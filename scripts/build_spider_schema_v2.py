#!/usr/bin/env python3
"""Build an enhanced Spider schema index with primary/foreign key context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def normalize_type(raw_type: str | None) -> str:
    value = (raw_type or "text").strip().upper()
    return value or "TEXT"


def column_lookup(db: dict[str, Any]) -> dict[int, dict[str, Any]]:
    table_names = db["table_names_original"]
    column_names = db["column_names_original"]
    column_types = db.get("column_types", [])
    lookup: dict[int, dict[str, Any]] = {}

    for column_idx, (table_idx, column_name) in enumerate(column_names):
        if table_idx < 0:
            continue
        lookup[column_idx] = {
            "index": column_idx,
            "table_index": table_idx,
            "table": table_names[table_idx],
            "column": column_name,
            "type": normalize_type(column_types[column_idx] if column_idx < len(column_types) else "text"),
        }
    return lookup


def build_schema_v1(db: dict[str, Any]) -> str:
    table_names = db["table_names_original"]
    columns = column_lookup(db)
    by_table: dict[int, list[dict[str, Any]]] = {idx: [] for idx in range(len(table_names))}
    for column in columns.values():
        by_table[column["table_index"]].append(column)

    statements: list[str] = []
    for table_idx, table_name in enumerate(table_names):
        defs = [f"{quote_ident(col['column'])} {col['type']}" for col in by_table.get(table_idx, [])]
        statements.append(f"CREATE TABLE {quote_ident(table_name)} ({', '.join(defs)});")
    return "\n".join(statements)


def build_relationships(db: dict[str, Any], columns: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []
    for pair in db.get("foreign_keys", []):
        if len(pair) != 2:
            continue
        from_idx, to_idx = pair
        from_col = columns.get(from_idx)
        to_col = columns.get(to_idx)
        if not from_col or not to_col:
            continue
        relationships.append(
            {
                "from_column_index": from_idx,
                "from_table": from_col["table"],
                "from_column": from_col["column"],
                "to_column_index": to_idx,
                "to_table": to_col["table"],
                "to_column": to_col["column"],
            }
        )
    return relationships


def build_schema_v2(db: dict[str, Any]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    table_names = db["table_names_original"]
    primary_key_indices = set(db.get("primary_keys", []))
    columns = column_lookup(db)
    relationships = build_relationships(db, columns)

    by_table: dict[int, list[dict[str, Any]]] = {idx: [] for idx in range(len(table_names))}
    fks_by_table: dict[str, list[dict[str, Any]]] = {name: [] for name in table_names}
    primary_keys: list[dict[str, Any]] = []

    for column in columns.values():
        by_table[column["table_index"]].append(column)
        if column["index"] in primary_key_indices:
            primary_keys.append(
                {
                    "column_index": column["index"],
                    "table": column["table"],
                    "column": column["column"],
                }
            )

    for relationship in relationships:
        fks_by_table.setdefault(relationship["from_table"], []).append(relationship)

    statements: list[str] = []
    for table_idx, table_name in enumerate(table_names):
        defs: list[str] = []
        for column in by_table.get(table_idx, []):
            suffix = " PRIMARY KEY" if column["index"] in primary_key_indices else ""
            defs.append(f"  {quote_ident(column['column'])} {column['type']}{suffix}")
        for relationship in fks_by_table.get(table_name, []):
            defs.append(
                "  FOREIGN KEY "
                f"({quote_ident(relationship['from_column'])}) "
                f"REFERENCES {quote_ident(relationship['to_table'])}"
                f"({quote_ident(relationship['to_column'])})"
            )

        if defs:
            statement = f"CREATE TABLE {quote_ident(table_name)} (\n" + ",\n".join(defs) + "\n);"
        else:
            statement = f"CREATE TABLE {quote_ident(table_name)} ();"
        statements.append(statement)

    if relationships:
        statements.append("-- Foreign key relationships:")
        for relationship in relationships:
            statements.append(
                "-- "
                f"{quote_ident(relationship['from_table'])}.{quote_ident(relationship['from_column'])} "
                "-> "
                f"{quote_ident(relationship['to_table'])}.{quote_ident(relationship['to_column'])}"
            )

    return "\n".join(statements), primary_keys, relationships


def build_index(tables: list[dict[str, Any]], database_dir: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for db in tables:
        db_id = db["db_id"]
        sqlite_path = database_dir / db_id / f"{db_id}.sqlite"
        schema_v2, primary_keys, foreign_keys = build_schema_v2(db)
        output[db_id] = {
            "db_id": db_id,
            "sqlite_path": str(sqlite_path),
            "sqlite_exists": sqlite_path.exists(),
            "schema": schema_v2,
            "schema_v1": build_schema_v1(db),
            "schema_v2": schema_v2,
            "schema_version": "spider_schema_v2_pk_fk",
            "table_names_original": db["table_names_original"],
            "column_names_original": db["column_names_original"],
            "column_types": db.get("column_types", []),
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables-json", default="/root/project/data/raw/spider/tables.json")
    parser.add_argument("--database-dir", default="/root/project/data/raw/spider/database")
    parser.add_argument("--output", default="/root/project/data/processed/spider_schema_v2.json")
    args = parser.parse_args()

    tables_path = Path(args.tables_json)
    database_dir = Path(args.database_dir)
    tables = json.loads(tables_path.read_text(encoding="utf-8"))
    index = build_index(tables, database_dir)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    missing = [db_id for db_id, item in index.items() if not item["sqlite_exists"]]
    schema_lengths = [len(item["schema_v2"]) for item in index.values()]
    fk_counts = [len(item["foreign_keys"]) for item in index.values()]
    pk_counts = [len(item["primary_keys"]) for item in index.values()]
    print(
        json.dumps(
            {
                "db_count": len(index),
                "missing_sqlite_count": len(missing),
                "missing_sqlite": missing[:20],
                "avg_schema_chars": round(mean(schema_lengths), 1) if schema_lengths else 0,
                "avg_primary_keys": round(mean(pk_counts), 2) if pk_counts else 0,
                "avg_foreign_keys": round(mean(fk_counts), 2) if fk_counts else 0,
                "output": str(output),
                "schema_version": "spider_schema_v2_pk_fk",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
