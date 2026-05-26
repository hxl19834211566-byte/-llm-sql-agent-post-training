#!/usr/bin/env python3
"""Build a high-confidence SQL DPO v3 split from DPO v1 pairs."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_QUOTAS = {
    "column_ownership_high": 110,
    "unnecessary_join_clean": 110,
    "missing_join_clean": 45,
    "simple_aggregation": 55,
    "join_aggregation": 35,
    "set_operation": 20,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def has_join(sql: str | None) -> bool:
    return bool(re.search(r"\bjoin\b", sql or "", flags=re.I))


def has_aggregation(sql: str | None) -> bool:
    norm = normalize_text(sql)
    return bool(re.search(r"\b(count|avg|sum|min|max)\s*\(", norm)) or " group by " in f" {norm} " or " having " in f" {norm} "


def has_set_operation(sql: str | None) -> bool:
    padded = f" {normalize_text(sql)} "
    return any(token in padded for token in [" intersect ", " except ", " union "])


def is_select(sql: str | None) -> bool:
    norm = normalize_text(sql)
    return norm.startswith(("select", "with"))


def classify_pair(row: dict[str, Any]) -> str | None:
    chosen = row.get("chosen") or ""
    rejected = row.get("rejected") or ""
    pred_error = normalize_text(row.get("pred_error"))
    pred_ok = bool(row.get("pred_exec_success"))
    chosen_join = has_join(chosen)
    rejected_join = has_join(rejected)
    chosen_agg = has_aggregation(chosen)
    rejected_agg = has_aggregation(rejected)
    chosen_set = has_set_operation(chosen)
    rejected_set = has_set_operation(rejected)

    if not chosen or not rejected or normalize_text(chosen) == normalize_text(rejected):
        return None
    if not is_select(chosen) or not is_select(rejected):
        return None
    if len(chosen) > 1400 or len(rejected) > 1400:
        return None

    if not pred_ok and ("no such column" in pred_error or "ambiguous column" in pred_error):
        return "column_ownership_high"
    if chosen_set or rejected_set:
        return "set_operation" if pred_ok else None
    if pred_ok and rejected_join and not chosen_join:
        return "unnecessary_join_clean"
    if pred_ok and chosen_join and not rejected_join:
        return "missing_join_clean"
    if pred_ok and (chosen_agg or rejected_agg):
        if chosen_join or rejected_join:
            return "join_aggregation"
        return "simple_aggregation"
    return None


def score_row(row: dict[str, Any], category: str) -> tuple[int, int, int]:
    chosen = row.get("chosen") or ""
    rejected = row.get("rejected") or ""
    length_gap = len(rejected) - len(chosen)
    category_bonus = {
        "column_ownership_high": int("no such column" in normalize_text(row.get("pred_error"))),
        "unnecessary_join_clean": int(has_join(rejected) and not has_join(chosen)),
        "missing_join_clean": int(has_join(chosen) and not has_join(rejected)),
        "simple_aggregation": int(has_aggregation(chosen) and has_aggregation(rejected)),
        "join_aggregation": int(has_join(chosen) and has_join(rejected)),
        "set_operation": int(has_set_operation(chosen) or has_set_operation(rejected)),
    }.get(category, 0)
    return (category_bonus, -abs(length_gap), -len(chosen))


def parse_quotas(raw: str | None) -> dict[str, int]:
    quotas = dict(DEFAULT_QUOTAS)
    if not raw:
        return quotas
    for item in raw.split(","):
        name, value = item.split("=", 1)
        quotas[name.strip()] = int(value)
    return quotas


def split_rows(rows: list[dict[str, Any]], heldout_size: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    heldout_n = min(max(0, heldout_size), max(0, len(shuffled) - 1))
    return shuffled[heldout_n:], shuffled[:heldout_n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--heldout-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--quotas", default=None)
    parser.add_argument("--heldout-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()

    quotas = parse_quotas(args.quotas)
    source_rows = read_jsonl(Path(args.input))
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped: Counter[str] = Counter()

    for row in source_rows:
        category = classify_pair(row)
        if not category:
            skipped["not_v3_high_confidence"] += 1
            continue
        buckets[category].append(row)

    selected: list[dict[str, Any]] = []
    for category, quota in quotas.items():
        rows = sorted(buckets.get(category, []), key=lambda row: score_row(row, category), reverse=True)
        for row in rows[:quota]:
            new_row = dict(row)
            new_row["dpo_v3_source_id"] = row.get("id")
            new_row["dpo_v3_category"] = category
            new_row["format_version"] = "dpo_v3_high_confidence_from_v1"
            selected.append(new_row)

    for index, row in enumerate(selected, start=1):
        row["id"] = f"dpo_v3_{row['dpo_v3_category']}_{index:06d}"

    train_rows, heldout_rows = split_rows(selected, args.heldout_size, args.seed)
    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.heldout_output), heldout_rows)

    summary = {
        "input": args.input,
        "source_rows": len(source_rows),
        "available_by_category": {name: len(rows) for name, rows in sorted(buckets.items())},
        "quotas": quotas,
        "selected_rows": len(selected),
        "selected_by_category": dict(Counter(row["dpo_v3_category"] for row in selected)),
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "skipped": dict(skipped),
        "format_version": "dpo_v3_high_confidence_from_v1",
        "note": "DPO v3 high-confidence mix guided by SFT v3 fine error analysis. Eval500 is used only to choose error categories, not as training rows.",
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
