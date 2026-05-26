#!/usr/bin/env python3
"""Train a lightweight SQL ORPO LoRA from SFT adapter and preference pairs."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_pair_rows(path: Path, max_samples: int | None = None) -> list[dict[str, str]]:
    rows = read_jsonl(path)
    if max_samples is not None:
        rows = rows[:max_samples]
    return [
        {
            "prompt": row["prompt"].rstrip() + "\n",
            "chosen": row["chosen"].strip(),
            "rejected": row["rejected"].strip(),
        }
        for row in rows
    ]


def encode_completion(
    tokenizer,
    prompt: str,
    completion: str,
    max_length: int,
    prompt_head_tokens: int,
) -> dict[str, list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    suffix = tokenizer.eos_token or ""
    completion_ids = tokenizer(completion + suffix, add_special_tokens=False).input_ids
    if not completion_ids:
        completion_ids = [tokenizer.eos_token_id]

    if len(completion_ids) >= max_length:
        completion_ids = completion_ids[: max_length - 1]

    prompt_budget = max_length - len(completion_ids)
    if len(prompt_ids) > prompt_budget:
        if prompt_budget > prompt_head_tokens:
            tail_len = prompt_budget - prompt_head_tokens
            prompt_ids = prompt_ids[:prompt_head_tokens] + prompt_ids[-tail_len:]
        else:
            prompt_ids = prompt_ids[-prompt_budget:]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class PairCollator:
    def __init__(self, tokenizer, max_length: int, prompt_head_tokens: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_head_tokens = prompt_head_tokens

    def _pad(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in features)
        input_ids = []
        attention_mask = []
        labels = []
        for item in features:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __call__(self, rows: list[dict[str, str]]) -> dict[str, dict[str, torch.Tensor]]:
        chosen = [
            encode_completion(
                self.tokenizer,
                row["prompt"],
                row["chosen"],
                self.max_length,
                self.prompt_head_tokens,
            )
            for row in rows
        ]
        rejected = [
            encode_completion(
                self.tokenizer,
                row["prompt"],
                row["rejected"],
                self.max_length,
                self.prompt_head_tokens,
            )
            for row in rows
        ]
        return {"chosen": self._pad(chosen), "rejected": self._pad(rejected)}


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def sequence_stats(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    logits = outputs.logits[:, :-1, :].float()
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~mask, 0)
    token_logps = torch.gather(F.log_softmax(logits, dim=-1), dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * mask
    token_counts = mask.sum(dim=-1).clamp(min=1)
    seq_logp = token_logps.sum(dim=-1) / token_counts
    nll = -seq_logp
    return nll, seq_logp, token_counts


def log_odds_from_avg_logp(avg_logp: torch.Tensor) -> torch.Tensor:
    stable_logp = torch.clamp(avg_logp, max=-1e-4)
    probability = torch.exp(stable_logp)
    return stable_logp - torch.log1p(-probability)


def orpo_loss(
    model,
    chosen_batch: dict[str, torch.Tensor],
    rejected_batch: dict[str, torch.Tensor],
    orpo_alpha: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    chosen_nll, chosen_logp, chosen_tokens = sequence_stats(model, chosen_batch)
    rejected_nll, rejected_logp, rejected_tokens = sequence_stats(model, rejected_batch)

    chosen_log_odds = log_odds_from_avg_logp(chosen_logp)
    rejected_log_odds = log_odds_from_avg_logp(rejected_logp)
    odds_margin = chosen_log_odds - rejected_log_odds
    preference_loss = -F.logsigmoid(odds_margin).mean()
    sft_loss = chosen_nll.mean()
    loss = sft_loss + orpo_alpha * preference_loss
    accuracy = (chosen_logp > rejected_logp).float().mean()

    metrics = {
        "loss": float(loss.detach().cpu()),
        "sft_loss": float(sft_loss.detach().cpu()),
        "preference_loss": float(preference_loss.detach().cpu()),
        "chosen_logp": float(chosen_logp.mean().detach().cpu()),
        "rejected_logp": float(rejected_logp.mean().detach().cpu()),
        "odds_margin": float(odds_margin.mean().detach().cpu()),
        "preference_accuracy": float(accuracy.detach().cpu()),
        "chosen_tokens": float(chosen_tokens.float().mean().detach().cpu()),
        "rejected_tokens": float(rejected_tokens.float().mean().detach().cpu()),
    }
    return loss, metrics


def mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted(items[0])
    return {key: sum(item[key] for item in items) / len(items) for key in keys}


@torch.no_grad()
def evaluate(model, dataloader: DataLoader, device: torch.device, orpo_alpha: float, max_batches: int | None = None) -> dict[str, float]:
    model.eval()
    metrics: list[dict[str, float]] = []
    for step, batch in enumerate(dataloader, start=1):
        chosen = move_batch(batch["chosen"], device)
        rejected = move_batch(batch["rejected"], device)
        _, batch_metrics = orpo_loss(model, chosen, rejected, orpo_alpha)
        metrics.append(batch_metrics)
        if max_batches is not None and step >= max_batches:
            break
    model.train()
    return mean_metrics(metrics)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sft-adapter-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--prompt-head-tokens", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--orpo-alpha", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = load_pair_rows(Path(args.train_file), args.max_train_samples)
    eval_rows = load_pair_rows(Path(args.eval_file), args.max_eval_samples) if args.eval_file else []

    config_payload = {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "model_path": args.model_path,
        "sft_adapter_path": args.sft_adapter_path,
        "output_dir": args.output_dir,
        "max_length": args.max_length,
        "orpo_alpha": args.orpo_alpha,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
    }
    print(json.dumps(config_payload, ensure_ascii=False, indent=2), flush=True)
    write_json(output_dir / "orpo_config.json", config_payload)

    collator = PairCollator(tokenizer, args.max_length, args.prompt_head_tokens)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_rows,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    eval_loader = (
        DataLoader(
            eval_rows,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        if eval_rows
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, args.sft_adapter_path, is_trainable=True)
    model.to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_update_steps = max(
        1,
        math.ceil(len(train_loader) * args.num_train_epochs / args.gradient_accumulation_steps),
    )

    def lr_scale(update_step: int) -> float:
        if args.warmup_steps <= 0:
            return 1.0
        return min(1.0, update_step / args.warmup_steps)

    train_metric_history: list[dict[str, float]] = []
    eval_metric_history: list[dict[str, Any]] = []
    global_step = 0
    update_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch_index in range(math.ceil(args.num_train_epochs)):
        for batch in train_loader:
            if global_step / max(1, len(train_loader)) >= args.num_train_epochs:
                break
            global_step += 1
            chosen = move_batch(batch["chosen"], device)
            rejected = move_batch(batch["rejected"], device)
            loss, metrics = orpo_loss(model, chosen, rejected, args.orpo_alpha)
            (loss / args.gradient_accumulation_steps).backward()
            train_metric_history.append(metrics)

            if global_step % args.gradient_accumulation_steps == 0:
                update_step += 1
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    args.max_grad_norm,
                )
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate * lr_scale(update_step)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if update_step % args.logging_steps == 0 or update_step == 1:
                    recent = mean_metrics(train_metric_history[-args.logging_steps :])
                    recent.update({"epoch": epoch_index + 1, "global_step": global_step, "update_step": update_step})
                    print(json.dumps(recent, ensure_ascii=False), flush=True)

                if eval_loader is not None and args.eval_steps > 0 and update_step % args.eval_steps == 0:
                    eval_metrics = evaluate(model, eval_loader, device, args.orpo_alpha, args.max_eval_batches)
                    eval_metrics.update({"global_step": global_step, "update_step": update_step})
                    eval_metric_history.append(eval_metrics)
                    print(json.dumps({"eval": eval_metrics}, ensure_ascii=False), flush=True)

            if update_step >= total_update_steps:
                break

    if global_step % args.gradient_accumulation_steps != 0:
        update_step += 1
        torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad],
            args.max_grad_norm,
        )
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * lr_scale(update_step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    final_train_metrics = mean_metrics(train_metric_history)
    final_eval_metrics = (
        evaluate(model, eval_loader, device, args.orpo_alpha, args.max_eval_batches) if eval_loader is not None else {}
    )
    summary = {
        **config_payload,
        "global_steps": global_step,
        "update_steps": update_step,
        "train_metrics": final_train_metrics,
        "eval_metrics": final_eval_metrics,
        "eval_history": eval_metric_history,
    }
    write_json(output_dir / "train_results.json", summary)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(json.dumps({"saved_checkpoint": args.output_dir, **summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
