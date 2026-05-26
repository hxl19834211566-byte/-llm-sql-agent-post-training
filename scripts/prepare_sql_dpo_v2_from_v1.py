#!/usr/bin/env python3
"""Build a conservative SQL DPO v2 split from DPO v1 pairs."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def has_join(sql: str) -> bool:
    return bool(re.search(r"\bjoin\b", sql or "", flags=re.IGNORECASE))


def is_high_confidence_v2(row: dict[str, Any]) -> bool:
    """Keep only obvious schema-linking failures for v2.

    DPO v1 regressed on eval500 because broad wrong-result pairs were too noisy.
    V2 keeps rejected SQL that failed execution from column ownership errors,
    so the preference direction is unambiguous.
    """
    if row.get("error_type") != "column_ownership":
        return False
    pred_error = normalize_text(row.get("pred_error"))
    if "no such column" not in pred_error and "ambiguous column" not in pred_error:
        return False
    if row.get("pred_exec_success") is True:
        return False
    chosen = str(row.get("chosen", "")).strip()
    rejected = str(row.get("rejected", "")).strip()
    if not chosen or not rejected or normalize_text(chosen) == normalize_text(rejected):
        return False
    if len(chosen) > 1200 or len(rejected) > 1200:
        return False
    return True


def score_row(row: dict[str, Any]) -> tuple[int, int, int]:
    """Prefer concise chosen SQL and rejected SQL with unnecessary joins."""
    chosen = str(row.get("chosen", ""))
    rejected = str(row.get("rejected", ""))
    join_delta = int(has_join(rejected) and not has_join(chosen))
    length_gap = max(0, len(rejected) - len(chosen))
    return (join_delta, -len(chosen), length_gap)


def split_rows(rows: list[dict[str, Any]], heldout_size: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    heldout_n = min(heldout_size, max(0, len(shuffled) - 1))
    return shuffled[heldout_n:], shuffled[:heldout_n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--heldout-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--max-pairs", type=int, default=280)
    parser.add_argument("--heldout-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_rows = read_jsonl(Path(args.input))
    selected = [row for row in source_rows if is_high_confidence_v2(row)]
    selected.sort(key=score_row, reverse=True)
    selected = selected[: args.max_pairs]
    for index, row in enumerate(selected, start=1):
        row["dpo_v2_source_id"] = row.get("id")
        row["id"] = f"dpo_v2_column_{index:06d}"
        row["format_version"] = "dpo_v2_column_ownership_conservative"

    train_rows, heldout_rows = split_rows(selected, args.heldout_size, args.seed)
    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.heldout_output), heldout_rows)

    summary = {
        "input": args.input,
        "source_rows": len(source_rows),
        "selected_rows": len(selected),
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "error_type_counts": dict(Counter(row.get("error_type") for row in selected)),
        "pred_error_counts": dict(
            Counter("ambiguous_column" if "ambiguous column" in normalize_text(row.get("pred_error")) else "no_such_column" for row in selected)
        ),
        "join_delta_rows": sum(has_join(row.get("rejected", "")) and not has_join(row.get("chosen", "")) for row in selected),
        "format_version": "dpo_v2_column_ownership_conservative",
        "note": "Conservative DPO v2 split from v1. Keeps only SFT v3 rejected SQL with explicit no-such-column or ambiguous-column execution failures.",
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
