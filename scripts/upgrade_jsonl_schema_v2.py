#!/usr/bin/env python3
"""Replace Spider JSONL row schemas with enhanced schema_v2 text."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    schema_index: dict[str, Any],
    format_version: str | None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    upgraded: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for row in rows:
        new_row = dict(row)
        db_id = new_row.get("db_id")
        schema_info = schema_index.get(db_id) if db_id else None
        if schema_info and schema_info.get("schema_v2"):
            old_schema = str(new_row.get("schema", ""))
            new_schema = str(schema_info["schema_v2"])
            new_row["schema_v1"] = old_schema
            new_row["schema"] = new_schema
            new_row["schema_version"] = schema_info.get("schema_version", "spider_schema_v2_pk_fk")
            new_row["primary_keys"] = schema_info.get("primary_keys", [])
            new_row["foreign_keys"] = schema_info.get("foreign_keys", [])
            if "prompt" in new_row:
                new_row["prompt"] = replace_schema_in_prompt(str(new_row["prompt"]), old_schema, new_schema)
            counts["upgraded"] += 1
        else:
            new_row.setdefault("schema_version", "unchanged")
            counts["unchanged_no_db_id_or_schema_v2"] += 1
        if format_version:
            new_row["format_version"] = format_version
        upgraded.append(new_row)

    return upgraded, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--schema-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--format-version", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    schema_path = Path(args.schema_index)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")

    rows = read_jsonl(input_path)
    schema_index = json.loads(schema_path.read_text(encoding="utf-8"))
    upgraded, counts = upgrade_rows(rows, schema_index, args.format_version)
    write_jsonl(output_path, upgraded)

    summary = {
        "input": str(input_path),
        "schema_index": str(schema_path),
        "output": str(output_path),
        "rows": len(rows),
        "counts": dict(counts),
        "format_version": args.format_version,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
