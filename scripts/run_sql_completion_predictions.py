#!/usr/bin/env python3
"""Run SQL predictions for completion-format SFT adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


INSTRUCTION = "You are a BI SQL agent. Write one read-only SQLite SQL query. Return only the SQL."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(row: dict[str, Any]) -> str:
    db_id = row.get("db_id")
    db_line = f"\nDatabase id: {db_id}" if db_id else ""
    return (
        f"{INSTRUCTION}\n"
        f"{db_line}\n"
        f"Schema:\n{row.get('schema') or '(schema not provided)'}\n\n"
        f"Question: {row['question']}\n\n"
        "SQL:"
    )


def clean_generation(text: str) -> str:
    text = text.strip()
    markers = [
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
    cutoff_positions = [text.find(marker) for marker in markers if marker in text]
    if cutoff_positions:
        text = text[: min(cutoff_positions)].strip()
    if ";" in text:
        text = text.split(";", 1)[0].strip() + ";"
    return text


def has_output_pollution(text: str) -> bool:
    pollution_markers = ["NdrFcShort", "<|im_start|>", "<|im_end|>", "\nQuestion:", "\nSchema:", "\nassistant", "\nuser"]
    return any(marker in text for marker in pollution_markers)


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int) -> tuple[str, str]:
    inputs = tokenizer(prompt + "\n", return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=False)
    return raw, clean_generation(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stream-write", action="store_true")
    args = parser.parse_args()

    eval_rows = read_jsonl(Path(args.eval_file))
    if args.limit is not None:
        eval_rows = eval_rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    output_path = Path(args.output_file)
    if args.stream_write:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    predictions: list[dict[str, Any]] = []
    polluted = 0
    for index, row in enumerate(eval_rows, start=1):
        prompt = build_prompt(row)
        raw_prediction, prediction = generate_one(model, tokenizer, prompt, args.max_new_tokens)
        polluted += int(has_output_pollution(raw_prediction))
        pred_row = {
            "id": row["id"],
            "source": row["source"],
            "db_id": row.get("db_id"),
            "question": row["question"],
            "prompt": prompt,
            "raw_prediction": raw_prediction.strip(),
            "prediction": prediction,
            "gold_sql": row.get("gold_sql"),
            "gold_result": row.get("gold_result"),
            "expected_tools": row.get("expected_tools"),
            "expected_args": row.get("expected_args"),
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "prompt_template": "completion_sql",
            "raw_output_polluted": has_output_pollution(raw_prediction),
        }
        if args.stream_write:
            append_jsonl(output_path, pred_row)
        else:
            predictions.append(pred_row)
        print(f"[{index}/{len(eval_rows)}] {row['id']} -> {prediction[:120]!r}", flush=True)

    if not args.stream_write:
        write_jsonl(output_path, predictions)
    print(
        json.dumps(
            {
                "saved": len(eval_rows) if args.stream_write else len(predictions),
                "output_file": args.output_file,
                "raw_output_pollution_count": polluted,
                "raw_output_pollution_rate": polluted / len(eval_rows) if eval_rows else 0.0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
