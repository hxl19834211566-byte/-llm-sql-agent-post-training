#!/usr/bin/env python3
"""Generate multiple SQL candidates for completion-format SFT adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


INSTRUCTION = "You are a BI SQL agent. Write one read-only SQLite SQL query. Return only the SQL."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
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
    pollution_markers = [
        "NdrFcShort",
        "<|im_start|>",
        "<|im_end|>",
        "\nQuestion:",
        "\nSchema:",
        "\nassistant",
        "\nuser",
    ]
    return any(marker in text for marker in pollution_markers)


def normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").lower().split())


def decode_candidates(tokenizer, output_ids: torch.Tensor, input_len: int) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for generated_ids in output_ids:
        raw = tokenizer.decode(generated_ids[input_len:], skip_special_tokens=False)
        cleaned = clean_generation(raw)
        key = normalize_sql(cleaned)
        if not cleaned or key in seen:
            continue
        seen.add(key)
        candidates.append((raw.strip(), cleaned))
    return candidates


def generate_candidates(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    sample_count: int,
    temperature: float,
    top_p: float,
) -> list[tuple[str, str, str]]:
    import torch

    inputs = tokenizer(prompt + "\n", return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    generated: list[tuple[str, str, str]] = []

    with torch.no_grad():
        greedy_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    for raw, cleaned in decode_candidates(tokenizer, greedy_ids, input_len):
        generated.append(("greedy", raw, cleaned))

    if sample_count > 0:
        with torch.no_grad():
            sample_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=sample_count,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        seen = {normalize_sql(item[2]) for item in generated}
        for raw, cleaned in decode_candidates(tokenizer, sample_ids, input_len):
            key = normalize_sql(cleaned)
            if key in seen:
                continue
            seen.add(key)
            generated.append(("sample", raw, cleaned))

    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.num_candidates < 1:
        raise ValueError("--num-candidates must be >= 1")

    eval_rows = read_jsonl(Path(args.eval_file))
    if args.limit is not None:
        eval_rows = eval_rows[: args.limit]

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    rows: list[dict[str, Any]] = []
    polluted = 0
    sample_count = max(0, args.num_candidates - 1)
    for index, row in enumerate(eval_rows, start=1):
        prompt = build_prompt(row)
        generated = generate_candidates(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            sample_count=sample_count,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        candidates = []
        for rank, (method, raw, prediction) in enumerate(generated[: args.num_candidates], start=1):
            raw_polluted = has_output_pollution(raw)
            polluted += int(raw_polluted)
            candidates.append(
                {
                    "rank": rank,
                    "method": method,
                    "raw_prediction": raw,
                    "prediction": prediction,
                    "raw_output_polluted": raw_polluted,
                }
            )

        rows.append(
            {
                "id": row["id"],
                "source": row["source"],
                "db_id": row.get("db_id"),
                "question": row["question"],
                "prompt": prompt,
                "candidates": candidates,
                "prediction": candidates[0]["prediction"] if candidates else "",
                "gold_sql": row.get("gold_sql"),
                "gold_result": row.get("gold_result"),
                "expected_tools": row.get("expected_tools"),
                "expected_args": row.get("expected_args"),
                "model_path": args.model_path,
                "adapter_path": args.adapter_path,
                "prompt_template": "completion_sql_candidates",
            }
        )
        preview = candidates[0]["prediction"][:120] if candidates else ""
        print(f"[{index}/{len(eval_rows)}] {row['id']} -> {len(candidates)} candidates; greedy={preview!r}", flush=True)

    write_jsonl(Path(args.output_file), rows)
    print(
        json.dumps(
            {
                "saved": len(rows),
                "output_file": args.output_file,
                "candidate_count_requested": args.num_candidates,
                "raw_output_pollution_count": polluted,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
