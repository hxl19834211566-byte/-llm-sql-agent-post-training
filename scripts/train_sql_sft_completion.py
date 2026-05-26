#!/usr/bin/env python3
"""Train LoRA SFT on prompt/completion SQL rows."""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed


IGNORE_INDEX = -100


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class CompletionSFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int):
        self.features: list[dict[str, list[int]]] = []
        self.skipped_no_label = 0
        self.truncated = 0
        for row in rows:
            feature = self.convert_row(row, tokenizer, max_length)
            if feature is None:
                self.skipped_no_label += 1
                continue
            self.features.append(feature)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.features[index]

    def convert_row(self, row: dict[str, Any], tokenizer, max_length: int) -> dict[str, list[int]] | None:
        prompt = row["prompt"].rstrip()
        completion = row["completion"].strip()
        if not completion.endswith(";"):
            completion = completion.rstrip(";").strip() + ";"

        prompt_ids = tokenizer(prompt + "\n", add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            completion_ids = completion_ids + [tokenizer.eos_token_id]

        input_ids = prompt_ids + completion_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids

        if len(input_ids) > max_length:
            self.truncated += 1
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]

        if all(label == IGNORE_INDEX for label in labels):
            return None

        return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


@dataclass
class DataCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [IGNORE_INDEX] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def training_args_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": True,
        "report_to": ["tensorboard"],
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "logging_first_step": True,
    }
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "steps"
    kwargs["eval_steps"] = args.eval_steps
    if "save_strategy" in params:
        kwargs["save_strategy"] = "steps"
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=50)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    rows = read_jsonl(Path(args.train_file))
    if args.max_train_samples is not None:
        rows = rows[: args.max_train_samples]

    eval_size = min(args.max_eval_samples, max(1, len(rows) // 10))
    eval_rows = rows[:eval_size]
    train_rows = rows[eval_size:]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = CompletionSFTDataset(train_rows, tokenizer, args.max_length)
    eval_dataset = CompletionSFTDataset(eval_rows, tokenizer, args.max_length)
    print(
        json.dumps(
            {
                "raw_train_rows": len(train_rows),
                "raw_eval_rows": len(eval_rows),
                "tokenized_train_rows": len(train_dataset),
                "tokenized_eval_rows": len(eval_dataset),
                "train_skipped_no_label": train_dataset.skipped_no_label,
                "eval_skipped_no_label": eval_dataset.skipped_no_label,
                "train_truncated": train_dataset.truncated,
                "eval_truncated": eval_dataset.truncated,
                "max_length": args.max_length,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args_kwargs(args)),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(tokenizer),
    )
    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    print("saved_checkpoint", args.output_dir)


if __name__ == "__main__":
    main()
