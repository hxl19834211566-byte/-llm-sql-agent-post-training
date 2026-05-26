#!/usr/bin/env python3
"""Train SQL DPO LoRA from an SFT adapter and DPO preference pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_dpo_dataset(path: Path, max_samples: int | None = None) -> Dataset:
    rows = read_jsonl(path)
    if max_samples is not None:
        rows = rows[:max_samples]
    compact_rows = [
        {
            "prompt": row["prompt"].rstrip() + "\n",
            "chosen": row["chosen"].strip(),
            "rejected": row["rejected"].strip(),
        }
        for row in rows
    ]
    return Dataset.from_list(compact_rows)


def load_sft_peft_model(model_path: str, adapter_path: str, trainable: bool):
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=trainable)
    if trainable:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    else:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sft-adapter-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dpo_dataset(Path(args.train_file), args.max_train_samples)
    eval_dataset = load_dpo_dataset(Path(args.eval_file), args.max_eval_samples) if args.eval_file else None
    print(
        json.dumps(
            {
                "train_rows": len(train_dataset),
                "eval_rows": len(eval_dataset) if eval_dataset is not None else 0,
                "model_path": args.model_path,
                "sft_adapter_path": args.sft_adapter_path,
                "output_dir": args.output_dir,
                "max_length": args.max_length,
                "beta": args.beta,
                "learning_rate": args.learning_rate,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    model = load_sft_peft_model(args.model_path, args.sft_adapter_path, trainable=True)
    ref_model = load_sft_peft_model(args.model_path, args.sft_adapter_path, trainable=False)
    model.print_trainable_parameters()

    dpo_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        beta=args.beta,
        max_length=args.max_length,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    metrics["eval_samples"] = len(eval_dataset) if eval_dataset is not None else 0
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    print("saved_checkpoint", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
