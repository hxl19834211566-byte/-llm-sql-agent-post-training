#!/usr/bin/env python3
"""Add rule-based SQLite value-linking hints to Spider JSONL prompts."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "airline",
    "airlines",
    "airport",
    "airports",
    "also",
    "among",
    "before",
    "below",
    "between",
    "cars",
    "city",
    "code",
    "count",
    "country",
    "countries",
    "different",
    "does",
    "each",
    "find",
    "first",
    "from",
    "give",
    "have",
    "many",
    "model",
    "name",
    "names",
    "number",
    "show",
    "that",
    "their",
    "there",
    "these",
    "those",
    "what",
    "when",
    "where",
    "which",
    "with",
    "whose",
}

GENERIC_VALUE_TOKENS = {
    "airline",
    "airlines",
    "airways",
    "airport",
    "airports",
    "city",
    "cities",
    "code",
    "country",
    "countries",
    "flight",
    "flights",
    "name",
    "names",
}


VALUE_ALIASES = {
    "european": ["Europe"],
    "europe": ["Europe"],
    "american": ["USA", "United States", "America"],
    "usa": ["USA", "United States"],
    "u.s.": ["USA", "United States"],
    "us": ["USA", "United States"],
    "french": ["France"],
    "france": ["France"],
    "german": ["Germany"],
    "germany": ["Germany"],
    "italian": ["Italy"],
    "italy": ["Italy"],
    "japanese": ["Japan"],
    "japan": ["Japan"],
    "english": ["England", "United Kingdom", "UK"],
    "british": ["United Kingdom", "UK", "England"],
}


TEXT_TYPE_MARKERS = ("CHAR", "CLOB", "TEXT", "VARCHAR", "STRING")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def normalize_text(text: Any) -> str:
    value = str(text or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def tokenize(text: str) -> list[str]:
    return [token for token in normalize_text(text).split() if token and token not in STOPWORDS]


def short_value(value: Any, max_chars: int = 80) -> str:
    text = str(value)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def extract_mentions(question: str, max_mentions: int) -> list[str]:
    mentions: list[str] = []

    def add(value: str) -> None:
        value = " ".join(value.strip().strip(".,;:!?()[]{}").split())
        if not value:
            return
        norm = normalize_text(value)
        if not norm or norm in STOPWORDS:
            return
        if value not in mentions:
            mentions.append(value)

    for match in re.finditer(r"['\"]([^'\"]+)['\"]", question):
        add(match.group(1))

    for match in re.finditer(r"\b(?:[A-Z][A-Za-z0-9]*|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9]*|[A-Z]{2,}))*\b", question):
        add(match.group(0))

    for match in re.finditer(r"\b\d{2,4}(?:\.\d+)?\b", question):
        add(match.group(0))

    words = re.findall(r"[A-Za-z0-9]+", question)
    for word in words:
        norm = normalize_text(word)
        for alias in VALUE_ALIASES.get(norm, []):
            add(alias)

    for size in (4, 3, 2):
        for idx in range(0, max(0, len(words) - size + 1)):
            phrase_words = words[idx : idx + size]
            content = [word for word in phrase_words if normalize_text(word) not in STOPWORDS]
            if len(content) >= max(1, size - 1):
                add(" ".join(phrase_words))

    for word in words:
        norm = normalize_text(word)
        if len(norm) >= 4 and norm not in STOPWORDS and norm not in GENERIC_VALUE_TOKENS:
            add(word)

    return mentions[:max_mentions]


def is_text_column(sqlite_type: str | None) -> bool:
    value = (sqlite_type or "").upper()
    return any(marker in value for marker in TEXT_TYPE_MARKERS) or not value


def list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [str(row[0]) for row in rows]


def table_columns(conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    rows = conn.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
    return [
        {
            "cid": row[0],
            "name": str(row[1]),
            "type": str(row[2] or ""),
            "notnull": bool(row[3]),
            "default": row[4],
            "pk": int(row[5] or 0),
        }
        for row in rows
    ]


def row_field_priority(column: str, matched_column: str) -> tuple[int, str]:
    lowered = column.lower()
    if column == matched_column:
        return (0, lowered)
    if lowered in {"id", "uid"} or lowered.endswith("_id") or lowered.endswith("id"):
        return (1, lowered)
    if any(token in lowered for token in ("abbrev", "code")):
        return (2, lowered)
    if any(token in lowered for token in ("name", "title", "model", "maker", "country", "continent")):
        return (3, lowered)
    return (9, lowered)


def format_sql_value(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)):
        return str(value)
    text = short_value(value)
    return '"' + text.replace('"', '\\"') + '"'


def format_same_row(row: dict[str, Any], matched_column: str, max_fields: int) -> str:
    fields: list[str] = []
    for column in sorted(row, key=lambda item: row_field_priority(item, matched_column)):
        value = row[column]
        if value is None or str(value).strip() == "":
            continue
        fields.append(f"{column}={format_sql_value(value)}")
        if len(fields) >= max_fields:
            break
    return "; ".join(fields)


def score_value_match(mention: str, value: Any) -> float:
    mention_norm = normalize_text(mention)
    value_norm = normalize_text(value)
    if not mention_norm or not value_norm:
        return 0.0
    if len(value_norm) <= 1 and value_norm != mention_norm:
        return 0.0

    if mention_norm == value_norm:
        return 100.0
    if len(mention_norm) >= 3 and mention_norm in value_norm:
        return 92.0
    if len(value_norm) >= 3 and value_norm in mention_norm:
        return 88.0

    mention_tokens = set(mention_norm.split())
    value_tokens = set(value_norm.split())
    if mention_tokens and value_tokens:
        overlap = len(mention_tokens & value_tokens)
        if overlap:
            overlap_tokens = mention_tokens & value_tokens
            distinctive_overlap = {token for token in overlap_tokens if token not in GENERIC_VALUE_TOKENS}
            if overlap >= 2 or distinctive_overlap:
                precision = overlap / len(value_tokens)
                recall = overlap / len(mention_tokens)
                token_score = 72.0 + 20.0 * min(precision, recall)
                if token_score >= 82.0:
                    return token_score

    ratio = difflib.SequenceMatcher(None, mention_norm, value_norm).ratio()
    if ratio >= 0.86:
        return 80.0 + 15.0 * ratio
    return 0.0


def load_db_metadata(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    metadata: dict[str, list[dict[str, Any]]] = {}
    conn.row_factory = sqlite3.Row
    for table in list_tables(conn):
        columns = table_columns(conn, table)
        text_columns = [column["name"] for column in columns if is_text_column(column["type"])]
        if text_columns:
            metadata[table] = [{"name": column} for column in text_columns]
    return metadata


def like_patterns(mention: str) -> list[str]:
    norm = normalize_text(mention)
    if not norm:
        return []
    patterns = [f"%{norm}%"]
    tokens = [token for token in norm.split() if len(token) >= 3]
    if len(tokens) >= 2:
        patterns.append("%" + "%".join(tokens) + "%")
    elif len(tokens) == 1:
        patterns.append(f"%{tokens[0]}%")
    return list(dict.fromkeys(patterns))


def query_value_rows(
    conn: sqlite3.Connection,
    metadata: dict[str, list[dict[str, Any]]],
    mentions: list[str],
    max_rows_per_query: int,
) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    value_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for mention in mentions:
        patterns = like_patterns(mention)
        if not patterns:
            continue
        for table, columns in metadata.items():
            for column_info in columns:
                column = column_info["name"]
                for pattern in patterns:
                    query = (
                        f"SELECT * FROM {quote_ident(table)} "
                        f"WHERE lower(CAST({quote_ident(column)} AS TEXT)) LIKE ? "
                        "LIMIT ?"
                    )
                    try:
                        rows = conn.execute(query, (pattern, max_rows_per_query)).fetchall()
                    except sqlite3.DatabaseError:
                        continue
                    for sqlite_row in rows:
                        row = dict(sqlite_row)
                        value = row.get(column)
                        if value is None or str(value).strip() == "":
                            continue
                        key = (table.lower(), column.lower(), normalize_text(value))
                        if key in seen:
                            continue
                        seen.add(key)
                        value_rows.append(
                            {
                                "table": table,
                                "column": column,
                                "value": value,
                                "row": row,
                            }
                        )
    return value_rows


def find_value_hints(
    question: str,
    value_rows: list[dict[str, Any]],
    max_mentions: int,
    max_hints: int,
    min_score: float,
    max_row_fields: int,
) -> list[dict[str, Any]]:
    mentions = extract_mentions(question, max_mentions=max_mentions)
    scored: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for mention in mentions:
        for item in value_rows:
            score = score_value_match(mention, item["value"])
            if score < min_score:
                continue
            key = (item["table"].lower(), item["column"].lower(), normalize_text(item["value"]))
            if key in seen:
                continue
            seen.add(key)
            scored.append(
                {
                    "mention": mention,
                    "table": item["table"],
                    "column": item["column"],
                    "value": item["value"],
                    "score": round(score, 3),
                    "same_row": format_same_row(item["row"], item["column"], max_fields=max_row_fields),
                }
            )

    scored.sort(key=lambda item: (-item["score"], item["table"].lower(), item["column"].lower()))
    return scored[:max_hints]


def find_value_hints_sqlite(
    conn: sqlite3.Connection,
    metadata: dict[str, list[dict[str, Any]]],
    question: str,
    max_mentions: int,
    max_hints: int,
    min_score: float,
    max_row_fields: int,
    max_rows_per_query: int,
) -> list[dict[str, Any]]:
    mentions = extract_mentions(question, max_mentions=max_mentions)
    value_rows = query_value_rows(
        conn=conn,
        metadata=metadata,
        mentions=mentions,
        max_rows_per_query=max_rows_per_query,
    )
    return find_value_hints(
        question=question,
        value_rows=value_rows,
        max_mentions=max_mentions,
        max_hints=max_hints,
        min_score=min_score,
        max_row_fields=max_row_fields,
    )


def render_value_hints(hints: list[dict[str, Any]]) -> str:
    if not hints:
        return ""

    lines = [
        "-- Value linking hints matched from the question:",
        "-- Use these only when they match the question. If a foreign key stores an entity id, use the same-row id.",
    ]
    for hint in hints:
        prefix = f"-- {hint['table']}.{hint['column']} = {format_sql_value(hint['value'])}"
        same_row = hint.get("same_row")
        if same_row:
            prefix += f"; same row: {same_row}"
        lines.append(prefix)
    return "\n".join(lines)


def augment_schema(schema: str, hints: list[dict[str, Any]]) -> str:
    hint_text = render_value_hints(hints)
    if not hint_text:
        return schema
    return schema.rstrip() + "\n\n" + hint_text


def replace_schema_in_prompt(prompt: str, old_schema: str, new_schema: str) -> str:
    if old_schema and old_schema in prompt:
        return prompt.replace(old_schema, new_schema, 1)

    schema_marker = "Schema:\n"
    question_marker = "\n\nQuestion:"
    start = prompt.find(schema_marker)
    end = prompt.find(question_marker, start + len(schema_marker))
    if start >= 0 and end >= 0:
        return prompt[: start + len(schema_marker)] + new_schema.strip() + prompt[end:]
    return prompt


def upgrade_rows(
    rows: list[dict[str, Any]],
    sqlite_root: Path,
    max_mentions: int,
    max_hints: int,
    min_score: float,
    max_rows_per_query: int,
    max_row_fields: int,
    format_version: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    upgraded: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    hints_by_db: Counter[str] = Counter()
    metadata_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for row in rows:
        new_row = dict(row)
        db_id = str(new_row.get("db_id") or "")
        schema = str(new_row.get("schema") or "")
        sqlite_path = sqlite_root / db_id / f"{db_id}.sqlite"

        if not db_id:
            counts["missing_db_id"] += 1
            hints: list[dict[str, Any]] = []
        elif not sqlite_path.exists():
            counts["missing_sqlite"] += 1
            hints = []
        else:
            with sqlite3.connect(sqlite_path) as conn:
                if db_id not in metadata_cache:
                    metadata_cache[db_id] = load_db_metadata(conn)
                hints = find_value_hints_sqlite(
                    conn=conn,
                    metadata=metadata_cache[db_id],
                    question=str(new_row.get("question") or ""),
                    max_mentions=max_mentions,
                    max_hints=max_hints,
                    min_score=min_score,
                    max_row_fields=max_row_fields,
                    max_rows_per_query=max_rows_per_query,
                )

        if hints:
            new_schema = augment_schema(schema, hints)
            new_row["schema_without_value_hints"] = schema
            new_row["schema"] = new_schema
            if "prompt" in new_row:
                new_row["prompt"] = replace_schema_in_prompt(str(new_row["prompt"]), schema, new_schema)
            counts["rows_with_hints"] += 1
            counts["hints_total"] += len(hints)
            hints_by_db[db_id] += len(hints)
        else:
            counts["rows_without_hints"] += 1

        new_row["value_linking_hints"] = hints
        new_row["value_linking_version"] = "rule_sqlite_value_linking_v1"
        new_row["format_version"] = format_version
        upgraded.append(new_row)

    summary = {
        "rows": len(rows),
        "counts": dict(counts),
        "hints_by_db_top20": hints_by_db.most_common(20),
        "cached_db_count": len(metadata_cache),
        "value_linking_version": "rule_sqlite_value_linking_v1",
    }
    return upgraded, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--sqlite-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--format-version", default="eval_schema_v2_value_linking_prompt_v1")
    parser.add_argument("--max-mentions", type=int, default=24)
    parser.add_argument("--max-hints", type=int, default=8)
    parser.add_argument("--min-score", type=float, default=82.0)
    parser.add_argument("--max-rows-per-query", type=int, default=50)
    parser.add_argument("--max-row-fields", type=int, default=6)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")

    rows = read_jsonl(input_path)
    upgraded, summary = upgrade_rows(
        rows=rows,
        sqlite_root=Path(args.sqlite_root),
        max_mentions=args.max_mentions,
        max_hints=args.max_hints,
        min_score=args.min_score,
        max_rows_per_query=args.max_rows_per_query,
        max_row_fields=args.max_row_fields,
        format_version=args.format_version,
    )
    write_jsonl(output_path, upgraded)

    summary.update(
        {
            "input": str(input_path),
            "output": str(output_path),
            "sqlite_root": args.sqlite_root,
            "format_version": args.format_version,
            "max_mentions": args.max_mentions,
            "max_hints": args.max_hints,
            "min_score": args.min_score,
            "max_rows_per_query": args.max_rows_per_query,
            "max_row_fields": args.max_row_fields,
        }
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
