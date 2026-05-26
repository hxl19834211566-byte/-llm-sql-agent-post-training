#!/usr/bin/env python3
"""Validate Spider schema_v2 index against tables.json and SQLite files."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def column_lookup(db: dict[str, Any]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for column_idx, (table_idx, column_name) in enumerate(db["column_names_original"]):
        if table_idx < 0:
            continue
        lookup[column_idx] = {
            "table_index": table_idx,
            "table": db["table_names_original"][table_idx],
            "column": column_name,
        }
    return lookup


def expected_primary_keys(db: dict[str, Any]) -> list[dict[str, Any]]:
    lookup = column_lookup(db)
    primary_keys: list[dict[str, Any]] = []
    for column_idx in db.get("primary_keys", []):
        column = lookup.get(column_idx)
        if column:
            primary_keys.append(
                {
                    "column_index": column_idx,
                    "table": column["table"],
                    "column": column["column"],
                }
            )
    return primary_keys


def expected_foreign_keys(db: dict[str, Any]) -> list[dict[str, Any]]:
    lookup = column_lookup(db)
    foreign_keys: list[dict[str, Any]] = []
    for pair in db.get("foreign_keys", []):
        if len(pair) != 2:
            continue
        from_idx, to_idx = pair
        from_col = lookup.get(from_idx)
        to_col = lookup.get(to_idx)
        if not from_col or not to_col:
            continue
        foreign_keys.append(
            {
                "from_column_index": from_idx,
                "from_table": from_col["table"],
                "from_column": from_col["column"],
                "to_column_index": to_idx,
                "to_table": to_col["table"],
                "to_column": to_col["column"],
            }
        )
    return foreign_keys


def sqlite_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sqlite_ddl_info(sqlite_path: Path, tables: list[str]) -> tuple[set[tuple[str, str]], set[tuple[str, str, str, str]], dict[str, set[str]]]:
    pk_cols: set[tuple[str, str]] = set()
    fk_edges: set[tuple[str, str, str, str]] = set()
    table_cols: dict[str, set[str]] = {}
    conn = sqlite3.connect(sqlite_path)
    try:
        sqlite_tables = {
            row[0]
            for row in conn.execute("select name from sqlite_master where type='table' and name not like 'sqlite_%'")
        }
        for table in tables:
            if table not in sqlite_tables:
                continue
            table_cols[table] = set()
            for col in conn.execute(f"PRAGMA table_info({sqlite_ident(table)})").fetchall():
                table_cols[table].add(col[1])
                if col[5]:
                    pk_cols.add((table, col[1]))
            for fk in conn.execute(f"PRAGMA foreign_key_list({sqlite_ident(table)})").fetchall():
                fk_edges.add((table, fk[3], fk[2], fk[4]))
    finally:
        conn.close()
    return pk_cols, fk_edges, table_cols


def validate(schema_index: dict[str, Any], tables: list[dict[str, Any]]) -> dict[str, Any]:
    source_by_db = {db["db_id"]: db for db in tables}
    issues: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    long_schemas: list[dict[str, Any]] = []

    for db_id, item in schema_index.items():
        stats["db"] += 1
        db = source_by_db.get(db_id)
        if not db:
            issues.append({"db_id": db_id, "type": "missing_in_tables_json"})
            continue

        required = [
            "db_id",
            "sqlite_path",
            "sqlite_exists",
            "schema",
            "schema_v1",
            "schema_v2",
            "schema_version",
            "table_names_original",
            "column_names_original",
            "column_types",
            "primary_keys",
            "foreign_keys",
        ]
        for key in required:
            if key not in item:
                issues.append({"db_id": db_id, "type": "missing_key", "key": key})

        if item.get("schema") != item.get("schema_v2"):
            issues.append({"db_id": db_id, "type": "schema_not_equal_schema_v2"})
        if item.get("schema_version") != "spider_schema_v2_pk_fk":
            issues.append({"db_id": db_id, "type": "unexpected_schema_version", "value": item.get("schema_version")})
        if item.get("table_names_original") != db.get("table_names_original"):
            issues.append({"db_id": db_id, "type": "table_names_mismatch"})
        if item.get("column_names_original") != db.get("column_names_original"):
            issues.append({"db_id": db_id, "type": "column_names_mismatch"})
        if item.get("column_types") != db.get("column_types"):
            issues.append({"db_id": db_id, "type": "column_types_mismatch"})

        expected_pk = sorted(expected_primary_keys(db), key=lambda row: row["column_index"])
        actual_pk = sorted(item.get("primary_keys", []), key=lambda row: row["column_index"])
        expected_fk = sorted(expected_foreign_keys(db), key=lambda row: (row["from_column_index"], row["to_column_index"]))
        actual_fk = sorted(item.get("foreign_keys", []), key=lambda row: (row["from_column_index"], row["to_column_index"]))
        if actual_pk != expected_pk:
            issues.append({"db_id": db_id, "type": "primary_keys_mismatch", "expected": expected_pk, "actual": actual_pk})
        if actual_fk != expected_fk:
            issues.append({"db_id": db_id, "type": "foreign_keys_mismatch", "expected": expected_fk, "actual": actual_fk})

        stats["tables"] += len(db.get("table_names_original", []))
        stats["columns"] += len([col for col in db.get("column_names_original", []) if col[0] >= 0])
        stats["primary_keys"] += len(actual_pk)
        stats["foreign_keys"] += len(actual_fk)

        schema_text = item.get("schema_v2", "")
        if len(schema_text) > 4000:
            long_schemas.append({"db_id": db_id, "chars": len(schema_text)})
        for table in db.get("table_names_original", []):
            if f"CREATE TABLE {quote_ident(table)}" not in schema_text:
                issues.append({"db_id": db_id, "type": "missing_create_table_text", "table": table})
        lookup = column_lookup(db)
        for column_idx, column in lookup.items():
            if quote_ident(column["column"]) not in schema_text:
                issues.append(
                    {"db_id": db_id, "type": "missing_column_text", "column_index": column_idx, "column": column}
                )
        for pk in actual_pk:
            pattern = re.escape(quote_ident(pk["column"])) + r"\s+[^,\n]+\s+PRIMARY KEY"
            if not re.search(pattern, schema_text):
                issues.append({"db_id": db_id, "type": "missing_primary_key_text", "primary_key": pk})
        for fk in actual_fk:
            fk_text = (
                f"FOREIGN KEY ({quote_ident(fk['from_column'])}) "
                f"REFERENCES {quote_ident(fk['to_table'])}({quote_ident(fk['to_column'])})"
            )
            if fk_text not in schema_text:
                issues.append({"db_id": db_id, "type": "missing_foreign_key_text", "foreign_key": fk})

        sqlite_path = Path(str(item.get("sqlite_path", "")))
        if sqlite_path.exists():
            sqlite_pk, sqlite_fk, sqlite_cols = sqlite_ddl_info(sqlite_path, db["table_names_original"])
            stats["sqlite_primary_keys"] += len(sqlite_pk)
            stats["sqlite_foreign_keys"] += len(sqlite_fk)
            v2_pk_pairs = {(pk["table"], pk["column"]) for pk in actual_pk}
            v2_fk_edges = {(fk["from_table"], fk["from_column"], fk["to_table"], fk["to_column"]) for fk in actual_fk}
            for table_idx, column_name in db.get("column_names_original", []):
                if table_idx < 0:
                    continue
                table = db["table_names_original"][table_idx]
                if table in sqlite_cols and column_name not in sqlite_cols[table]:
                    issues.append({"db_id": db_id, "type": "missing_sqlite_column", "table": table, "column": column_name})
            if sqlite_pk and sqlite_pk != v2_pk_pairs:
                issues.append(
                    {
                        "db_id": db_id,
                        "type": "sqlite_primary_key_differs_from_tables_json",
                        "sqlite": sorted(sqlite_pk),
                        "schema_v2": sorted(v2_pk_pairs),
                    }
                )
            if sqlite_fk and sqlite_fk != v2_fk_edges:
                issues.append(
                    {
                        "db_id": db_id,
                        "type": "sqlite_foreign_key_differs_from_tables_json",
                        "sqlite": sorted(sqlite_fk),
                        "schema_v2": sorted(v2_fk_edges),
                    }
                )
        else:
            issues.append({"db_id": db_id, "type": "missing_sqlite_path", "sqlite_path": str(sqlite_path)})

    return {
        "stats": dict(stats),
        "issue_count": len(issues),
        "issues": issues,
        "long_schema_count": len(long_schemas),
        "long_schema_top10": sorted(long_schemas, key=lambda row: row["chars"], reverse=True)[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema-index", required=True)
    parser.add_argument("--tables-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    schema_index = read_json(Path(args.schema_index))
    tables = read_json(Path(args.tables_json))
    result = validate(schema_index, tables)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    printable = dict(result)
    printable["issues"] = result["issues"][:30]
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
