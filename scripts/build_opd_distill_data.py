#!/usr/bin/env python3
"""Build OPD completion data from execution-correct teacher SQL predictions."""

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


def ensure_semicolon(sql: str) -> str:
    sql = sql.strip()
    if not sql:
        return sql
    return sql if sql.endswith(";") else f"{sql};"


def normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").lower().split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-prompts", required=True)
    parser.add_argument("--teacher-predictions", required=True)
    parser.add_argument("--teacher-exec-eval", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--heldout-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--heldout-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-name", default="qwen3_8b")
    parser.add_argument("--only-teacher-correct", action="store_true", default=True)
    args = parser.parse_args()

    source_rows = {row["id"]: row for row in read_jsonl(Path(args.source_prompts))}
    prediction_rows = {row["id"]: row for row in read_jsonl(Path(args.teacher_predictions))}
    eval_rows = read_jsonl(Path(args.teacher_exec_eval))

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    counts = {
        "source_prompts": len(source_rows),
        "teacher_predictions": len(prediction_rows),
        "teacher_eval_rows": len(eval_rows),
        "missing_source": 0,
        "missing_prediction": 0,
        "teacher_not_correct": 0,
        "empty_teacher_sql": 0,
        "duplicate_prompt_completion": 0,
        "selected": 0,
    }

    for eval_row in eval_rows:
        row_id = eval_row.get("id")
        source = source_rows.get(row_id)
        prediction = prediction_rows.get(row_id)
        if not source:
            counts["missing_source"] += 1
            continue
        if not prediction:
            counts["missing_prediction"] += 1
            continue
        if args.only_teacher_correct and not eval_row.get("exec_match"):
            counts["teacher_not_correct"] += 1
            continue

        teacher_sql = ensure_semicolon(str(eval_row.get("pred_sql") or prediction.get("prediction") or ""))
        if not teacher_sql:
            counts["empty_teacher_sql"] += 1
            continue

        key = (str(source["prompt"]).rstrip(), normalize_sql(teacher_sql))
        if key in seen:
            counts["duplicate_prompt_completion"] += 1
            continue
        seen.add(key)

        row = dict(source)
        row.update(
            {
                "id": f"opd_v1_{row_id}",
                "source": "opd_teacher_sql",
                "base_source": source.get("source"),
                "distill_source_id": row_id,
                "completion": teacher_sql,
                "teacher_sql": teacher_sql,
                "teacher_prediction": prediction.get("prediction"),
                "teacher_model_path": prediction.get("model_path"),
                "teacher_name": args.teacher_name,
                "teacher_exec_match": bool(eval_row.get("exec_match")),
                "teacher_pred_exec_success": bool(eval_row.get("pred_exec_success")),
                "teacher_gold_exec_success": bool(eval_row.get("gold_exec_success")),
                "gold_sql": source.get("gold_sql"),
                "format_version": "opd_v1_teacher_correct_schema_v2_completion",
            }
        )
        selected.append(row)

    random.Random(args.seed).shuffle(selected)
    heldout_size = min(args.heldout_size, max(0, len(selected) // 5))
    heldout_rows = selected[:heldout_size]
    train_rows = selected[heldout_size:]

    counts["selected"] = len(selected)
    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.heldout_output), heldout_rows)

    summary = {
        "source_prompts": args.source_prompts,
        "teacher_predictions": args.teacher_predictions,
        "teacher_exec_eval": args.teacher_exec_eval,
        "train_output": args.train_output,
        "heldout_output": args.heldout_output,
        "teacher_name": args.teacher_name,
        "heldout_size_requested": args.heldout_size,
        "heldout_size_actual": len(heldout_rows),
        "train_size": len(train_rows),
        "seed": args.seed,
        "counts": counts,
    }
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
