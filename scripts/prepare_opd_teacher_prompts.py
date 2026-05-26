#!/usr/bin/env python3
"""Prepare Spider train prompts for OPD teacher generation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("db_id", "")).strip().lower(),
        " ".join(str(row.get("question", "")).strip().lower().split()),
        " ".join(str(row.get("gold_sql") or row.get("query") or "").strip().lower().split()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--source", default="xlangai/spider")
    parser.add_argument("--exclude-eval-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    excluded: set[tuple[str, str, str]] = set()
    if args.exclude_eval_file:
        excluded = {row_key(row) for row in read_jsonl(Path(args.exclude_eval_file))}

    selected: list[dict[str, Any]] = []
    counts = {
        "input_rows": len(rows),
        "excluded_eval_overlap": 0,
        "missing_required_fields": 0,
        "wrong_source": 0,
        "selected": 0,
    }

    for row in rows:
        if row.get("source") != args.source:
            counts["wrong_source"] += 1
            continue
        if row_key(row) in excluded:
            counts["excluded_eval_overlap"] += 1
            continue
        if not row.get("prompt") or not row.get("db_id") or not row.get("gold_sql"):
            counts["missing_required_fields"] += 1
            continue

        new_row = dict(row)
        new_row["format_version"] = "opd_v1_teacher_prompt_schema_v2"
        selected.append(new_row)

    if args.shuffle:
        random.Random(args.seed).shuffle(selected)
    if args.limit is not None:
        selected = selected[: args.limit]

    counts["selected"] = len(selected)
    write_jsonl(Path(args.output), selected)
    summary = {
        "input": args.input,
        "output": args.output,
        "source": args.source,
        "exclude_eval_file": args.exclude_eval_file,
        "limit": args.limit,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "counts": counts,
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
