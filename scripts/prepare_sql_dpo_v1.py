#!/usr/bin/env python3
"""Build SQL DPO v1 pairs from Spider train and SFT v3 mistakes."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


INSTRUCTION = "You are a BI SQL agent. Write one read-only SQLite SQL query. Return only the SQL."
FORMAT_VERSION = "dpo_v1_sft_v3_mistakes"
DEFAULT_ERROR_QUOTAS = {
    "column_ownership": 280,
    "wrong_result": 260,
    "unnecessary_join": 160,
    "join_path_error": 160,
    "aggregation_error": 100,
    "set_operation_error": 40,
}


def normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
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


def clean_sql(sql: str) -> str:
    sql = re.sub(r"\s+", " ", sql or "").strip()
    sql = sql.rstrip(";").strip()
    return f"{sql};" if sql else ""


def extract_sql(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    match = re.search(r"\b(select|with)\b", text, flags=re.IGNORECASE)
    if match:
        text = text[match.start() :].strip()
    cutoff_markers = [
        "NdrFcShort",
        "<|im_end|>",
        "<|endoftext|>",
        "<|im_start|>",
        "\nQuestion:",
        "\nSchema:",
        "\nDatabase id:",
        "\nassistant",
        "\nuser",
    ]
    cutoff_positions = [text.find(marker) for marker in cutoff_markers if marker in text]
    if cutoff_positions:
        text = text[: min(cutoff_positions)].strip()
    if ";" in text:
        text = text.split(";", 1)[0].strip() + ";"
    return text.strip()


def build_prompt(db_id: str, schema: str, question: str) -> str:
    return (
        f"{INSTRUCTION}\n"
        f"\nDatabase id: {db_id}\n"
        f"Schema:\n{schema.strip()}\n\n"
        f"Question: {question.strip()}\n\n"
        "SQL:"
    )


def has_order_by(sql: str) -> bool:
    return bool(re.search(r"\border\s+by\b", sql, flags=re.IGNORECASE))


def normalize_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def normalize_rows(rows: list[tuple[Any, ...]], ordered: bool) -> list[list[Any]]:
    normalized = [[normalize_value(value) for value in row] for row in rows]
    if not ordered:
        normalized = sorted(normalized, key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True))
    return normalized


def execute_sql(sqlite_path: str, sql: str, timeout_sec: float) -> tuple[bool, list[list[Any]] | None, str | None]:
    if not read_only_sql(sql):
        return False, None, "not_read_only_select"
    uri = f"file:{sqlite_path}?mode=ro"
    conn = None
    try:
        deadline = time.monotonic() + timeout_sec
        conn = sqlite3.connect(uri, uri=True, timeout=timeout_sec)
        conn.execute("PRAGMA query_only = ON")
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
        rows = conn.execute(sql).fetchall()
        return True, [list(row) for row in rows], None
    except Exception as exc:  # noqa: BLE001
        return False, None, f"{type(exc).__name__}: {exc}"
    finally:
        if conn is not None:
            conn.close()


def is_exec_match(
    sqlite_path: str,
    gold_sql: str,
    pred_sql: str,
    timeout_sec: float,
) -> tuple[bool, dict[str, Any]]:
    gold_ok, gold_rows, gold_error = execute_sql(sqlite_path, gold_sql, timeout_sec)
    pred_ok, pred_rows, pred_error = execute_sql(sqlite_path, pred_sql, timeout_sec)
    ordered = has_order_by(gold_sql)
    gold_norm = normalize_rows([tuple(row) for row in (gold_rows or [])], ordered) if gold_ok else None
    pred_norm = normalize_rows([tuple(row) for row in (pred_rows or [])], ordered) if pred_ok else None
    return bool(gold_ok and pred_ok and gold_norm == pred_norm), {
        "gold_exec_success": gold_ok,
        "pred_exec_success": pred_ok,
        "gold_error": gold_error,
        "pred_error": pred_error,
        "ordered_compare": ordered,
    }


def has_join(sql: str) -> bool:
    return bool(re.search(r"\bjoin\b", sql, flags=re.IGNORECASE))


def has_aggregation(sql: str) -> bool:
    normalized = normalize_text(sql)
    return bool(re.search(r"\b(count|avg|sum|min|max)\s*\(", normalized)) or " group by " in f" {normalized} "


def has_set_operation(sql: str) -> bool:
    padded = f" {normalize_text(sql)} "
    return any(token in padded for token in [" intersect ", " except ", " union "])


def classify_error(gold_sql: str, pred_sql: str, pred_error: str | None) -> str:
    error_text = normalize_text(pred_error)
    if "no such column" in error_text or "ambiguous column" in error_text:
        return "column_ownership"
    if has_set_operation(gold_sql) or has_set_operation(pred_sql):
        return "set_operation_error"
    if has_aggregation(gold_sql) or has_aggregation(pred_sql):
        return "aggregation_error"
    if has_join(pred_sql) and not has_join(gold_sql):
        return "unnecessary_join"
    if has_join(pred_sql) or has_join(gold_sql):
        return "join_path_error"
    return "wrong_result"


def eval_overlap_keys(eval_rows: list[dict[str, Any]]) -> tuple[set[tuple[str | None, str]], set[tuple[str | None, str, str]]]:
    db_question = {(row.get("db_id"), normalize_text(row.get("question"))) for row in eval_rows}
    db_question_sql = {
        (row.get("db_id"), normalize_text(row.get("question")), normalize_text(row.get("gold_sql")))
        for row in eval_rows
    }
    return db_question, db_question_sql


def build_candidates(
    schema_index: dict[str, Any],
    eval_rows: list[dict[str, Any]],
    max_scan: int,
    max_prompt_chars: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    eval_db_question, eval_db_question_sql = eval_overlap_keys(eval_rows)
    skipped: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    seen_db_question: set[tuple[str | None, str]] = set()
    ds = load_dataset("xlangai/spider", split="train", streaming=True)

    for scanned, raw in enumerate(ds, start=1):
        if scanned > max_scan:
            break
        row = dict(raw)
        db_id = row.get("db_id")
        question = str(row.get("question", "")).strip()
        gold_sql = clean_sql(str(row.get("query", "")).strip())
        db_info = schema_index.get(db_id)
        if not db_info:
            skipped["missing_schema"] += 1
            continue
        if not question or not read_only_sql(gold_sql):
            skipped["bad_question_or_sql"] += 1
            continue
        db_question_key = (db_id, normalize_text(question))
        db_question_sql_key = (db_id, normalize_text(question), normalize_text(gold_sql))
        if db_question_key in eval_db_question or db_question_sql_key in eval_db_question_sql:
            skipped["eval_overlap"] += 1
            continue
        if db_question_key in seen_db_question:
            skipped["duplicate_question"] += 1
            continue
        prompt = build_prompt(db_id, db_info["schema"], question)
        if len(prompt) > max_prompt_chars:
            skipped["prompt_too_long"] += 1
            continue
        sqlite_path = db_info["sqlite_path"]
        if not Path(sqlite_path).exists():
            skipped["missing_sqlite"] += 1
            continue
        candidates.append(
            {
                "db_id": db_id,
                "schema": db_info["schema"],
                "sqlite_path": sqlite_path,
                "question": question,
                "gold_sql": gold_sql,
                "prompt": prompt,
                "candidate_index": scanned,
            }
        )
        seen_db_question.add(db_question_key)
    return candidates, skipped


def load_model(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    return model, tokenizer


def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[tuple[str, str]]:
    encoded = tokenizer([prompt.rstrip() + "\n" for prompt in prompts], return_tensors="pt", padding=True).to(model.device)
    input_len = encoded["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    out: list[tuple[str, str]] = []
    for row_ids in output_ids:
        raw = tokenizer.decode(row_ids[input_len:], skip_special_tokens=False).strip()
        out.append((raw, extract_sql(raw)))
    return out


def scaled_quotas(target_pairs: int) -> dict[str, int]:
    total = sum(DEFAULT_ERROR_QUOTAS.values())
    quotas = {name: round(value * target_pairs / total) for name, value in DEFAULT_ERROR_QUOTAS.items()}
    delta = target_pairs - sum(quotas.values())
    names = list(quotas)
    for idx in range(abs(delta)):
        name = names[idx % len(names)]
        quotas[name] += 1 if delta > 0 else -1
    return quotas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/root/project")
    parser.add_argument("--model-path", default="/root/project/hf_cache/models/Qwen3-4B-Base")
    parser.add_argument("--adapter-path", default="/root/project/checkpoints/sft/sql_sft_v3_qwen3_4b")
    parser.add_argument("--schema-index", default=None)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--target-pairs", type=int, default=1000)
    parser.add_argument("--max-scan", type=int, default=8000)
    parser.add_argument("--max-prompt-chars", type=int, default=18000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root)
    schema_index_path = Path(args.schema_index) if args.schema_index else root / "data/processed/spider_schema.json"
    eval_file = Path(args.eval_file) if args.eval_file else root / "data/eval/spider_schema_eval_500.jsonl"
    output = Path(args.output) if args.output else root / "data/dpo/sql_dpo_v1_1000.jsonl"
    summary_output = (
        Path(args.summary_output) if args.summary_output else root / "logs/sql_dpo_v1_1000_summary.json"
    )

    if args.overwrite and output.exists():
        output.unlink()

    existing_rows = read_jsonl(output)
    existing_keys = {(row.get("db_id"), normalize_text(row.get("question"))) for row in existing_rows}
    counts: Counter[str] = Counter(row.get("error_type", "unknown") for row in existing_rows)
    quotas = scaled_quotas(args.target_pairs)

    schema_index = json.loads(schema_index_path.read_text(encoding="utf-8"))
    eval_rows = read_jsonl(eval_file)
    candidates, skipped_candidates = build_candidates(
        schema_index=schema_index,
        eval_rows=eval_rows,
        max_scan=args.max_scan,
        max_prompt_chars=args.max_prompt_chars,
    )
    print(
        json.dumps(
            {
                "event": "candidates_ready",
                "candidates": len(candidates),
                "skipped_candidates": dict(skipped_candidates),
                "existing_pairs": len(existing_rows),
                "target_pairs": args.target_pairs,
                "quotas": quotas,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    model, tokenizer = load_model(args)
    accepted = len(existing_rows)
    processed = 0
    skipped_pairs: Counter[str] = Counter()

    for start in range(0, len(candidates), args.batch_size):
        if accepted >= args.target_pairs:
            break
        batch = [row for row in candidates[start : start + args.batch_size] if (row["db_id"], normalize_text(row["question"])) not in existing_keys]
        if not batch:
            continue
        generations = generate_batch(model, tokenizer, [row["prompt"] for row in batch], args.max_new_tokens)
        new_rows: list[dict[str, Any]] = []
        for row, (raw_prediction, pred_sql) in zip(batch, generations):
            processed += 1
            db_question_key = (row["db_id"], normalize_text(row["question"]))
            if db_question_key in existing_keys:
                skipped_pairs["duplicate_output_key"] += 1
                continue
            gold_match, exec_info = is_exec_match(row["sqlite_path"], row["gold_sql"], pred_sql, args.timeout_sec)
            if not exec_info["gold_exec_success"]:
                skipped_pairs["gold_exec_failed"] += 1
                continue
            if gold_match:
                skipped_pairs["prediction_correct"] += 1
                continue
            if not read_only_sql(pred_sql):
                skipped_pairs["prediction_not_read_only"] += 1
                continue
            if normalize_text(row["gold_sql"]) == normalize_text(pred_sql):
                skipped_pairs["chosen_rejected_same"] += 1
                continue

            error_type = classify_error(row["gold_sql"], pred_sql, exec_info.get("pred_error"))
            if counts[error_type] >= quotas.get(error_type, 0):
                skipped_pairs[f"quota_full_{error_type}"] += 1
                continue

            pair_id = f"dpo_spider_train_{accepted + 1:06d}"
            dpo_row = {
                "id": pair_id,
                "source": "xlangai/spider_train",
                "db_id": row["db_id"],
                "schema": row["schema"],
                "question": row["question"],
                "prompt": row["prompt"],
                "chosen": row["gold_sql"],
                "rejected": pred_sql,
                "error_type": error_type,
                "chosen_source": "gold_sql",
                "rejected_source": "sft_v3_prediction",
                "raw_rejected": raw_prediction,
                "candidate_index": row["candidate_index"],
                "pred_exec_success": exec_info["pred_exec_success"],
                "pred_error": exec_info["pred_error"],
                "format_version": FORMAT_VERSION,
            }
            new_rows.append(dpo_row)
            existing_keys.add(db_question_key)
            counts[error_type] += 1
            accepted += 1
            if accepted >= args.target_pairs:
                break
        if new_rows:
            append_jsonl(output, new_rows)
        if processed % max(args.batch_size * 10, 40) == 0 or new_rows:
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "processed": processed,
                        "accepted": accepted,
                        "counts": dict(counts),
                        "skipped_pairs": dict(skipped_pairs),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = {
        "output": str(output),
        "rows": accepted,
        "target_pairs": args.target_pairs,
        "candidate_count": len(candidates),
        "processed_candidates": processed,
        "error_type_counts": dict(counts),
        "quotas": quotas,
        "skipped_candidates": dict(skipped_candidates),
        "skipped_pairs": dict(skipped_pairs),
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "schema_index": str(schema_index_path),
        "eval_file": str(eval_file),
        "format_version": FORMAT_VERSION,
        "note": "DPO v1 pairs from Spider train. Eval500 db_id+question overlaps are excluded. Chosen is gold SQL; rejected is SFT v3 mistake.",
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
