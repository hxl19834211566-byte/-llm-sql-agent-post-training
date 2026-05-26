#!/usr/bin/env python3
"""Validate and split SQL DPO v1 preference pairs."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = ["prompt", "chosen", "rejected", "error_type", "db_id", "question"]


def normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_only_sql(sql: str) -> bool:
    normalized = normalize_text(sql)
    if not normalized.startswith(("select", "with")):
        return False
    padded = f" {normalized} "
    blocked = [
        " delete ",
        " drop ",
        " update ",
        " alter ",
        " insert ",
        " create ",
        " truncate ",
        " replace ",
        " merge ",
        " grant ",
        " revoke ",
        " attach ",
        " detach ",
        " pragma ",
        " vacuum ",
        " call ",
        " exec ",
    ]
    return not any(token in padded for token in blocked)


def eval_overlap_keys(eval_rows: list[dict[str, Any]]) -> set[tuple[str | None, str]]:
    return {(row.get("db_id"), normalize_text(row.get("question"))) for row in eval_rows}


def validate_rows(rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> tuple[Counter[str], dict[str, Any]]:
    issue_counts: Counter[str] = Counter()
    eval_keys = eval_overlap_keys(eval_rows)
    seen_db_question: set[tuple[str | None, str]] = set()
    seen_pair: set[tuple[str, str, str]] = set()
    error_type_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for row in rows:
        missing = [field for field in REQUIRED_FIELDS if not row.get(field)]
        if missing:
            issue_counts["missing_required_field"] += 1
        db_question_key = (row.get("db_id"), normalize_text(row.get("question")))
        if db_question_key in eval_keys:
            issue_counts["eval500_db_question_overlap"] += 1
        if db_question_key in seen_db_question:
            issue_counts["duplicate_db_question"] += 1
        seen_db_question.add(db_question_key)

        chosen = str(row.get("chosen", "")).strip()
        rejected = str(row.get("rejected", "")).strip()
        if normalize_text(chosen) == normalize_text(rejected):
            issue_counts["chosen_rejected_same"] += 1
        if not read_only_sql(chosen):
            issue_counts["chosen_not_read_only"] += 1
        if not read_only_sql(rejected):
            issue_counts["rejected_not_read_only"] += 1
        if any(marker in rejected for marker in ["NdrFcShort", "<|im_start|>", "<|im_end|>", "\nQuestion:", "\nSchema:"]):
            issue_counts["rejected_output_pollution"] += 1

        pair_key = (normalize_text(row.get("prompt")), normalize_text(chosen), normalize_text(rejected))
        if pair_key in seen_pair:
            issue_counts["duplicate_prompt_pair"] += 1
        seen_pair.add(pair_key)

        error_type_counts[str(row.get("error_type", "unknown"))] += 1
        source_counts[str(row.get("source", "unknown"))] += 1

    summary = {
        "rows": len(rows),
        "issue_counts": dict(issue_counts),
        "error_type_counts": dict(error_type_counts),
        "source_counts": dict(source_counts),
    }
    return issue_counts, summary


def split_rows(rows: list[dict[str, Any]], heldout_size: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    heldout_n = min(max(0, heldout_size), max(0, len(shuffled) - 1))
    heldout_rows = shuffled[:heldout_n]
    train_rows = shuffled[heldout_n:]
    return train_rows, heldout_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--heldout-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--heldout-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.dpo_file))
    eval_rows = read_jsonl(Path(args.eval_file))
    issue_counts, summary = validate_rows(rows, eval_rows)
    blocking_issues = {
        "missing_required_field",
        "eval500_db_question_overlap",
        "chosen_rejected_same",
        "chosen_not_read_only",
        "rejected_not_read_only",
        "duplicate_prompt_pair",
    }
    blocking_count = sum(issue_counts.get(name, 0) for name in blocking_issues)

    train_rows, heldout_rows = split_rows(rows, args.heldout_size, args.seed)
    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.heldout_output), heldout_rows)

    summary.update(
        {
            "dpo_file": args.dpo_file,
            "eval_file": args.eval_file,
            "train_output": args.train_output,
            "heldout_output": args.heldout_output,
            "train_rows": len(train_rows),
            "heldout_rows": len(heldout_rows),
            "heldout_size_requested": args.heldout_size,
            "seed": args.seed,
            "blocking_issue_count": blocking_count,
            "status": "pass" if blocking_count == 0 else "fail",
        }
    )
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if blocking_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
