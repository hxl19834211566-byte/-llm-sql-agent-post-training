#!/usr/bin/env python3
"""Train a small GRPO-style SQL LoRA from the SFT v3 adapter.

This implementation is optimized for small validation runs:
- one prompt is sampled into a group of SQL rollouts;
- each SQL is executed against Spider SQLite;
- execution-match is the main reward;
- group-normalized rewards become advantages;
- a small SFT anchor keeps the model close to the gold SQL behavior.

The goal is to test whether the candidate oracle can exceed the current
value-linking n20 upper bound, not to replace the final reranker directly.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sqlite3
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any


INSTRUCTION = "You are a BI SQL agent. Write one read-only SQLite SQL query. Return only the SQL."
IGNORE_INDEX = -100
SQL_KEYWORDS = {
    "as",
    "by",
    "cross",
    "except",
    "full",
    "group",
    "having",
    "inner",
    "intersect",
    "join",
    "left",
    "limit",
    "on",
    "order",
    "right",
    "union",
    "where",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().rstrip(";").lower().split())


def clean_generation(text: str) -> str:
    text = (text or "").strip()
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
    match = re.search(r"\b(select|with)\b", text, flags=re.IGNORECASE)
    if match:
        text = text[match.start() :].strip()
    if ";" in text:
        text = text.split(";", 1)[0].strip() + ";"
    return text


def normalize_completion_sql(sql: str) -> str:
    sql = clean_generation(sql)
    if sql and not sql.endswith(";"):
        sql = sql.rstrip(";").strip() + ";"
    return sql


def build_prompt(row: dict[str, Any]) -> str:
    if row.get("prompt"):
        return str(row["prompt"]).rstrip() + "\n"
    db_id = row.get("db_id")
    db_line = f"\nDatabase id: {db_id}" if db_id else ""
    return (
        f"{INSTRUCTION}\n"
        f"{db_line}\n"
        f"Schema:\n{row.get('schema') or '(schema not provided)'}\n\n"
        f"Question: {row['question']}\n\n"
        "SQL:\n"
    )


def gold_sql(row: dict[str, Any]) -> str:
    value = str(row.get("completion") or row.get("gold_sql") or "").strip()
    if value and not value.endswith(";"):
        value = value.rstrip(";").strip() + ";"
    return value


def sqlite_path(sqlite_root: Path, db_id: str) -> Path:
    return sqlite_root / db_id / f"{db_id}.sqlite"


def collect_usable_rows(
    rows: list[dict[str, Any]],
    sqlite_root: Path,
    max_samples: int | None,
    timeout_sec: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_kept: list[dict[str, Any]] = []
    counts = {
        "raw_rows": len(rows),
        "scanned_rows": 0,
        "missing_db_id": 0,
        "missing_sqlite": 0,
        "missing_gold_sql": 0,
        "gold_exec_failed": 0,
        "usable_rows_before_limit": 0,
        "sampled_usable_rows": 0,
    }
    gold_fail_examples: list[dict[str, Any]] = []
    usable_by_db: Counter[str] = Counter()
    for row in rows:
        counts["scanned_rows"] += 1
        db_id = str(row.get("db_id") or "")
        if not db_id:
            counts["missing_db_id"] += 1
            continue
        db_path = sqlite_path(sqlite_root, db_id)
        if not db_path.exists():
            counts["missing_sqlite"] += 1
            continue
        gold = gold_sql(row)
        if not gold:
            counts["missing_gold_sql"] += 1
            continue
        gold_ok, _, gold_error = execute_sql(db_path, gold, timeout_sec)
        if not gold_ok:
            counts["gold_exec_failed"] += 1
            if len(gold_fail_examples) < 5:
                gold_fail_examples.append(
                    {
                        "id": row.get("id"),
                        "db_id": db_id,
                        "gold_sql": gold,
                        "gold_error": gold_error,
                    }
                )
            continue
        all_kept.append(row)
        usable_by_db[db_id] += 1

    sampled = list(all_kept)
    random.Random(seed).shuffle(sampled)
    if max_samples is not None:
        sampled = sampled[:max_samples]

    counts["usable_rows_before_limit"] = len(all_kept)
    counts["sampled_usable_rows"] = len(sampled)
    sampled_by_db = Counter(str(row.get("db_id") or "") for row in sampled)
    return sampled, {
        "counts": counts,
        "usable_by_db_top20": usable_by_db.most_common(20),
        "sampled_by_db_top20": sampled_by_db.most_common(20),
        "gold_fail_examples": gold_fail_examples,
    }


def read_only_sql(sql: str) -> bool:
    normalized = re.sub(r"\s+", " ", sql).strip().lower()
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
    ]
    return not any(token in padded for token in blocked)


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


def execute_sql(path: Path, sql: str, timeout_sec: float) -> tuple[bool, list[list[Any]] | None, str | None]:
    if not read_only_sql(sql):
        return False, None, "not_read_only_select"
    conn = None
    try:
        deadline = time.monotonic() + timeout_sec
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout_sec)
        conn.execute("PRAGMA query_only = ON")
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
        rows = conn.execute(sql.rstrip(";")).fetchall()
        return True, [list(row) for row in rows], None
    except Exception as exc:  # noqa: BLE001
        return False, None, f"{type(exc).__name__}: {exc}"
    finally:
        if conn is not None:
            conn.close()


def parse_schema_tables(schema: str) -> dict[str, set[str]]:
    tables: dict[str, set[str]] = {}
    pattern = re.compile(r'CREATE\s+TABLE\s+"?([A-Za-z_][\w]*)"?\s*\((.*?)(?:\n\);|\);)', re.IGNORECASE | re.DOTALL)
    for match in pattern.finditer(schema or ""):
        table = match.group(1).lower()
        body = match.group(2)
        columns: set[str] = set()
        for line in body.splitlines():
            stripped = line.strip().rstrip(",")
            if not stripped or stripped.upper().startswith(("FOREIGN KEY", "PRIMARY KEY", "UNIQUE", "CONSTRAINT")):
                continue
            col_match = re.match(r'"?([A-Za-z_][\w]*)"?\s+', stripped)
            if col_match:
                columns.add(col_match.group(1).lower())
        tables[table] = columns
    return tables


def strip_ident(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] in {'"', "`", "["}:
        return value[1:-1].lower()
    return value.lower()


def schema_issues(sql: str, row: dict[str, Any]) -> list[str]:
    tables = parse_schema_tables(str(row.get("schema") or ""))
    if not tables:
        return []

    issues: list[str] = []
    alias_to_table: dict[str, str] = {}
    table_pattern = re.compile(
        r"\b(?:from|join)\s+([`\"\[]?[A-Za-z_][\w]*[`\"\]]?)(?:\s+(?:as\s+)?([A-Za-z_][\w]*))?",
        re.IGNORECASE,
    )
    for match in table_pattern.finditer(sql):
        table = strip_ident(match.group(1))
        alias_token = strip_ident(match.group(2) or "")
        alias = table if not alias_token or alias_token in SQL_KEYWORDS else alias_token
        if table not in tables:
            issues.append(f"unknown_table:{table}")
            continue
        alias_to_table[alias] = table
        alias_to_table[table] = table

    for match in re.finditer(r'([A-Za-z_][\w]*|"[^"]+"|`[^`]+`|\[[^\]]+\])\s*\.\s*([A-Za-z_][\w]*|"[^"]+"|`[^`]+`|\[[^\]]+\])', sql):
        owner = strip_ident(match.group(1))
        column = strip_ident(match.group(2))
        table = alias_to_table.get(owner)
        if not table:
            continue
        if column not in tables.get(table, set()):
            issues.append(f"unknown_column:{owner}.{column}")
    return issues


def intent_reward(question: str, sql: str) -> float:
    q = question.lower()
    s = sql.lower()
    reward = 0.0

    def wants(words: tuple[str, ...]) -> bool:
        return any(word in q for word in words)

    if wants(("how many", "number of", "count of", "total number")):
        reward += 0.35 if re.search(r"\bcount\s*\(", s) else -0.35
    if wants(("average", "avg ")):
        reward += 0.25 if re.search(r"\bavg\s*\(", s) else -0.25
    if wants(("maximum", "max ", "largest", "highest")):
        reward += 0.2 if re.search(r"\bmax\s*\(", s) or " order by " in f" {s} " else -0.15
    if wants(("minimum", "min ", "smallest", "lowest", "fewest")):
        reward += 0.2 if re.search(r"\bmin\s*\(", s) or " order by " in f" {s} " else -0.15
    if wants(("most", "least", "oldest", "youngest", "earliest", "latest")):
        reward += 0.25 if " order by " in f" {s} " else -0.2
    if wants(("each", "per ", " by ")):
        reward += 0.25 if " group by " in f" {s} " else 0.0
    if wants(("different", "distinctive", "distinct ")):
        reward += 0.2 if " distinct " in f" {s} " else -0.1
    if wants(("not ", " no ", " except ", " without ")):
        reward += 0.2 if any(token in s for token in (" not ", " except ", " not in ", "!=")) else 0.0
    return reward


def value_linking_reward(row: dict[str, Any], sql: str) -> float:
    hints = row.get("value_linking_hints") or []
    if not hints:
        return 0.0
    lowered = sql.lower()
    matched = 0
    for hint in hints:
        table = str(hint.get("table") or "").lower()
        column = str(hint.get("column") or "").lower()
        value = str(hint.get("value") or "").lower()
        if value and value in lowered:
            matched += 1
            continue
        if table and column and table in lowered and column in lowered:
            matched += 1
    if matched:
        return min(0.6, 0.25 * matched)
    return -0.15


def reward_sql(row: dict[str, Any], db_path: Path, prediction: str, timeout_sec: float) -> dict[str, Any]:
    sql = clean_generation(prediction)
    gold = gold_sql(row)
    gold_ok, gold_rows, gold_error = execute_sql(db_path, gold, timeout_sec)
    return reward_sql_with_gold(row, db_path, prediction, timeout_sec, gold_ok, gold_rows, gold_error)


def reward_sql_with_gold(
    row: dict[str, Any],
    db_path: Path,
    prediction: str,
    timeout_sec: float,
    gold_ok: bool,
    gold_rows: list[list[Any]] | None,
    gold_error: str | None,
) -> dict[str, Any]:
    sql = clean_generation(prediction)
    gold = gold_sql(row)
    pred_ok, pred_rows, pred_error = execute_sql(db_path, sql, timeout_sec)
    ordered = has_order_by(gold)
    gold_norm = normalize_rows([tuple(item) for item in (gold_rows or [])], ordered) if gold_ok else None
    pred_norm = normalize_rows([tuple(item) for item in (pred_rows or [])], ordered) if pred_ok else None
    exec_match = bool(gold_ok and pred_ok and gold_norm == pred_norm)
    issues = schema_issues(sql, row)

    reward = 0.0
    if exec_match:
        reward += 5.0
    if pred_ok:
        reward += 0.5
    else:
        reward -= 2.0
    if read_only_sql(sql):
        reward += 0.2
    else:
        reward -= 3.0
    if issues:
        reward -= min(2.0, 0.5 * len(issues))
    else:
        reward += 0.5
    if pred_ok and gold_ok and pred_norm == [] and gold_norm not in (None, []):
        reward -= 0.6
    reward += intent_reward(str(row.get("question") or ""), sql)
    reward += value_linking_reward(row, sql)
    reward = max(-4.0, min(8.0, reward))

    return {
        "sql": sql,
        "reward": reward,
        "exec_match": exec_match,
        "pred_exec_success": pred_ok,
        "pred_error": pred_error,
        "gold_exec_success": gold_ok,
        "gold_error": gold_error,
        "schema_issues": issues,
    }


def truncate_prompt_ids(input_ids: list[int], max_prompt_tokens: int, prompt_head_tokens: int) -> list[int]:
    if len(input_ids) <= max_prompt_tokens:
        return input_ids
    if max_prompt_tokens <= prompt_head_tokens:
        return input_ids[-max_prompt_tokens:]
    tail_len = max_prompt_tokens - prompt_head_tokens
    return input_ids[:prompt_head_tokens] + input_ids[-tail_len:]


def trim_continuation(ids: list[int], eos_token_id: int | None, pad_token_id: int | None) -> list[int]:
    trimmed: list[int] = []
    for token_id in ids:
        if pad_token_id is not None and token_id == pad_token_id:
            break
        trimmed.append(token_id)
        if eos_token_id is not None and token_id == eos_token_id:
            break
    return trimmed


def pad_features(features: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, Any]:
    import torch

    max_len = max(len(item["input_ids"]) for item in features)
    input_ids = []
    labels = []
    attention_mask = []
    for item in features:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        labels.append(item["labels"] + [IGNORE_INDEX] * pad_len)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def sequence_logp(model, batch: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    import torch.nn.functional as F

    seq_logps = []
    token_counts = []
    for row_idx in range(batch["input_ids"].shape[0]):
        row_len = int(batch["attention_mask"][row_idx].sum().detach().cpu())
        input_ids = batch["input_ids"][row_idx : row_idx + 1, :row_len]
        attention_mask = batch["attention_mask"][row_idx : row_idx + 1, :row_len]
        labels = batch["labels"][row_idx : row_idx + 1, :row_len]
        target_count = int(labels.ne(IGNORE_INDEX).sum().detach().cpu())
        if target_count <= 0:
            seq_logps.append(input_ids.new_tensor(0.0, dtype=torch.float32))
            token_counts.append(input_ids.new_tensor(0, dtype=torch.long))
            continue

        # Labels are constructed as prompt IGNOREs followed by a contiguous SQL
        # completion suffix. Keep only the final completion positions plus the
        # preceding position needed to predict the first completion token.
        logits_to_keep = min(row_len, target_count + 1)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=logits_to_keep,
        )
        logits = outputs.logits[:, :-1, :].float()
        target_labels = labels[:, -target_count:]
        token_logps = torch.gather(
            F.log_softmax(logits, dim=-1),
            dim=-1,
            index=target_labels.unsqueeze(-1),
        ).squeeze(-1)
        seq_logps.append(token_logps.sum(dim=-1).squeeze(0) / max(1, target_count))
        token_counts.append(input_ids.new_tensor(target_count, dtype=torch.long))

    return torch.stack(seq_logps), torch.stack(token_counts)


def sft_feature(tokenizer, row: dict[str, Any], max_length: int, prompt_head_tokens: int) -> dict[str, list[int]]:
    prompt_ids = tokenizer(build_prompt(row), add_special_tokens=False).input_ids
    completion = gold_sql(row)
    completion_ids = tokenizer(completion, add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None:
        completion_ids = completion_ids + [tokenizer.eos_token_id]
    if len(completion_ids) >= max_length:
        completion_ids = completion_ids[: max_length - 1]
    prompt_budget = max(1, max_length - len(completion_ids))
    prompt_ids = truncate_prompt_ids(prompt_ids, prompt_budget, prompt_head_tokens)
    input_ids = prompt_ids + completion_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids
    return {"input_ids": input_ids, "labels": labels}


def rollout_feature(
    tokenizer,
    prompt_ids: list[int],
    sql: str,
    max_length: int,
) -> dict[str, list[int]]:
    completion = normalize_completion_sql(sql)
    completion_ids = tokenizer(completion, add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None:
        completion_ids = completion_ids + [tokenizer.eos_token_id]
    if len(completion_ids) >= max_length:
        completion_ids = completion_ids[: max_length - 1]
    prompt_budget = max(1, max_length - len(completion_ids))
    trimmed_prompt_ids = prompt_ids[-prompt_budget:]
    return {
        "input_ids": trimmed_prompt_ids + completion_ids,
        "labels": [IGNORE_INDEX] * len(trimmed_prompt_ids) + completion_ids,
    }


def generate_group(
    model,
    tokenizer,
    row: dict[str, Any],
    max_prompt_tokens: int,
    prompt_head_tokens: int,
    max_new_tokens: int,
    num_generations: int,
    temperature: float,
    top_p: float,
) -> tuple[list[dict[str, Any]], int]:
    import torch

    prompt_ids = tokenizer(build_prompt(row), add_special_tokens=False).input_ids
    prompt_ids = truncate_prompt_ids(prompt_ids, max_prompt_tokens, prompt_head_tokens)
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_generations,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for generated_ids in outputs.detach().cpu().tolist():
        continuation = trim_continuation(
            generated_ids[len(prompt_ids) :],
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        )
        if not continuation:
            continue
        raw = tokenizer.decode(continuation, skip_special_tokens=False)
        cleaned = clean_generation(raw)
        key = normalize_sql(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "raw": raw.strip(),
                "prediction": cleaned,
                "feature": rollout_feature(tokenizer, prompt_ids, cleaned, max_prompt_tokens + max_new_tokens),
            }
        )
    return items, len(prompt_ids)


def grpo_step(
    model,
    tokenizer,
    row: dict[str, Any],
    db_path: Path,
    args: argparse.Namespace,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch

    model.eval()
    group, prompt_tokens = generate_group(
        model=model,
        tokenizer=tokenizer,
        row=row,
        max_prompt_tokens=args.max_prompt_tokens,
        prompt_head_tokens=args.prompt_head_tokens,
        max_new_tokens=args.max_new_tokens,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    model.train()

    if not group:
        sft_batch = move_batch(pad_features([sft_feature(tokenizer, row, args.max_length, args.prompt_head_tokens)], tokenizer.pad_token_id), device)
        sft_logp, _ = sequence_logp(model, sft_batch)
        loss = -sft_logp.mean() * args.sft_anchor_coef
        return loss, {
            "skipped_group": 1.0,
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "policy_loss": 0.0,
            "sft_loss": float((-sft_logp.mean()).detach().cpu()),
            "group_size": 0,
            "prompt_tokens": prompt_tokens,
        }

    gold = gold_sql(row)
    gold_ok, gold_rows, gold_error = execute_sql(db_path, gold, args.timeout_sec)
    if not gold_ok:
        sft_batch = move_batch(pad_features([sft_feature(tokenizer, row, args.max_length, args.prompt_head_tokens)], tokenizer.pad_token_id), device)
        sft_logp, _ = sequence_logp(model, sft_batch)
        loss = -sft_logp.mean() * args.sft_anchor_coef
        return loss, {
            "skipped_group": 1.0,
            "gold_exec_failed": 1.0,
            "gold_error": gold_error,
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "policy_loss": 0.0,
            "sft_loss": float((-sft_logp.mean()).detach().cpu()),
            "group_size": len(group),
            "prompt_tokens": prompt_tokens,
        }

    reward_items = [
        reward_sql_with_gold(row, db_path, item["prediction"], args.timeout_sec, gold_ok, gold_rows, gold_error)
        for item in group
    ]
    rewards = torch.tensor([item["reward"] for item in reward_items], dtype=torch.float32, device=device)
    reward_mean = rewards.mean()
    reward_std = rewards.std(unbiased=False)
    advantages = rewards - reward_mean
    if float(reward_std.detach().cpu()) > 1e-6:
        advantages = advantages / (reward_std + 1e-6)
    else:
        advantages = torch.zeros_like(advantages)

    rollout_batch = move_batch(pad_features([item["feature"] for item in group], tokenizer.pad_token_id), device)
    rollout_logp, rollout_tokens = sequence_logp(model, rollout_batch)
    policy_loss = -(advantages.detach() * rollout_logp).mean()

    sft_batch = move_batch(pad_features([sft_feature(tokenizer, row, args.max_length, args.prompt_head_tokens)], tokenizer.pad_token_id), device)
    sft_logp, sft_tokens = sequence_logp(model, sft_batch)
    sft_loss = -sft_logp.mean()
    loss = policy_loss + args.sft_anchor_coef * sft_loss

    best_index = int(torch.argmax(rewards).detach().cpu())
    metrics = {
        "skipped_group": 0.0,
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "sft_loss": float(sft_loss.detach().cpu()),
        "reward_mean": float(reward_mean.detach().cpu()),
        "reward_max": float(rewards.max().detach().cpu()),
        "reward_min": float(rewards.min().detach().cpu()),
        "reward_std": float(reward_std.detach().cpu()),
        "advantage_abs_mean": float(advantages.abs().mean().detach().cpu()),
        "group_size": len(group),
        "prompt_tokens": prompt_tokens,
        "rollout_tokens": float(rollout_tokens.float().mean().detach().cpu()),
        "sft_tokens": float(sft_tokens.float().mean().detach().cpu()),
        "group_exec_match": float(any(item["exec_match"] for item in reward_items)),
        "group_exec_success": float(any(item["pred_exec_success"] for item in reward_items)),
        "best_reward": float(reward_items[best_index]["reward"]),
        "best_sql": reward_items[best_index]["sql"],
        "best_exec_match": bool(reward_items[best_index]["exec_match"]),
        "rollouts": [
            {
                "sql": reward_item["sql"],
                "reward": reward_item["reward"],
                "exec_match": reward_item["exec_match"],
                "pred_exec_success": reward_item["pred_exec_success"],
                "schema_issue_count": len(reward_item["schema_issues"]),
            }
            for reward_item in reward_items
        ],
    }
    return loss, metrics


def reward_probe(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    sqlite_root: Path,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    probe_rows = rows[: max(1, args.probe_samples)]
    probe_path = output_dir / "reward_probe.jsonl"
    if probe_path.exists():
        probe_path.unlink()

    summaries: list[dict[str, Any]] = []
    model.eval()
    for row in probe_rows:
        db_path = sqlite_path(sqlite_root, str(row["db_id"]))
        group, prompt_tokens = generate_group(
            model=model,
            tokenizer=tokenizer,
            row=row,
            max_prompt_tokens=args.max_prompt_tokens,
            prompt_head_tokens=args.prompt_head_tokens,
            max_new_tokens=args.max_new_tokens,
            num_generations=args.num_generations,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        gold = gold_sql(row)
        gold_ok, gold_rows, gold_error = execute_sql(db_path, gold, args.timeout_sec)
        rewards = [
            reward_sql_with_gold(row, db_path, item["prediction"], args.timeout_sec, gold_ok, gold_rows, gold_error)
            for item in group
        ]
        best_reward = max((item["reward"] for item in rewards), default=0.0)
        summary = {
            "id": row.get("id"),
            "db_id": row.get("db_id"),
            "question": row.get("question"),
            "gold_sql": gold,
            "gold_exec_success": gold_ok,
            "gold_error": gold_error,
            "prompt_tokens": prompt_tokens,
            "group_size": len(group),
            "best_reward": best_reward,
            "group_exec_match": any(item["exec_match"] for item in rewards),
            "group_exec_success": any(item["pred_exec_success"] for item in rewards),
            "rollouts": rewards,
        }
        append_jsonl(probe_path, summary)
        summaries.append(summary)
    model.train()
    return {
        "probe_samples": len(summaries),
        "probe_path": str(probe_path),
        "avg_group_size": sum(item["group_size"] for item in summaries) / len(summaries) if summaries else 0.0,
        "group_exec_match": sum(1 for item in summaries if item["group_exec_match"]),
        "group_exec_success": sum(1 for item in summaries if item["group_exec_success"]),
        "gold_exec_success": sum(1 for item in summaries if item["gold_exec_success"]),
    }


def mean_numeric(metrics: list[dict[str, Any]]) -> dict[str, float]:
    if not metrics:
        return {}
    numeric_keys = sorted({
        key
        for item in metrics
        for key, value in item.items()
        if isinstance(value, (int, float, bool)) and key not in {"best_exec_match"}
    })
    return {
        key: sum(float(item.get(key, 0.0)) for item in metrics) / len(metrics)
        for key in numeric_keys
    }


def looks_like_eval_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return "/data/eval/" in normalized or "eval_500" in normalized or "eval500" in normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--sft-adapter-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--sqlite-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train-samples", type=int, default=800)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=1536)
    parser.add_argument("--max-length", type=int, default=1664)
    parser.add_argument("--prompt-head-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--sft-anchor-coef", type=float, default=0.05)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--rollout-log-steps", type=int, default=10)
    parser.add_argument("--min-usable-rows", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--probe-samples", type=int, default=2)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--allow-eval-train", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if Path(args.sft_adapter_path).resolve(strict=False) == output_dir.resolve(strict=False):
        raise ValueError("--output-dir must not be the same path as --sft-adapter-path.")
    if looks_like_eval_path(args.train_file) and not args.allow_eval_train:
        raise ValueError(
            "Refusing to train on an eval-looking file. "
            "Use a train/grpo file, or pass --allow-eval-train only for an intentional debug run."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = read_jsonl(Path(args.train_file))
    rows, preflight = collect_usable_rows(
        all_rows,
        Path(args.sqlite_root),
        args.max_train_samples,
        args.timeout_sec,
        args.seed,
    )
    write_json(output_dir / "preflight_summary.json", preflight)
    print(json.dumps({"preflight": preflight}, ensure_ascii=False, indent=2), flush=True)
    if len(rows) < args.min_usable_rows:
        raise ValueError(
            f"Only {len(rows)} usable training rows found; "
            f"required at least {args.min_usable_rows}. See preflight_summary.json."
        )
    if args.preflight_only:
        print(json.dumps({"preflight_only": True, "usable_rows": len(rows)}, ensure_ascii=False), flush=True)
        return

    config = {
        "route": "grpo_v1_from_sft_v3_schema_v2_value_linking",
        "model_path": args.model_path,
        "sft_adapter_path": args.sft_adapter_path,
        "train_file": args.train_file,
        "sqlite_root": args.sqlite_root,
        "output_dir": args.output_dir,
        "raw_train_rows": len(all_rows),
        "usable_train_rows": len(rows),
        "max_steps": args.max_steps,
        "num_generations": args.num_generations,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "sft_anchor_coef": args.sft_anchor_coef,
        "seed": args.seed,
        "preflight": preflight,
    }
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)
    write_json(output_dir / "grpo_config.json", config)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    random.seed(args.seed)
    random.shuffle(rows)
    rollout_log_path = output_dir / "rollout_samples.jsonl"
    if rollout_log_path.exists():
        rollout_log_path.unlink()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = PeftModel.from_pretrained(model, args.sft_adapter_path, is_trainable=True)
    model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.train()

    probe_summary = reward_probe(model, tokenizer, rows, Path(args.sqlite_root), args, output_dir)
    write_json(output_dir / "reward_probe_summary.json", probe_summary)
    print(json.dumps({"reward_probe": probe_summary}, ensure_ascii=False, indent=2), flush=True)
    if args.probe_only:
        print(json.dumps({"probe_only": True, **probe_summary}, ensure_ascii=False), flush=True)
        return

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer.zero_grad(set_to_none=True)

    def lr_scale(update_step: int) -> float:
        if args.warmup_steps <= 0:
            return 1.0
        return min(1.0, update_step / args.warmup_steps)

    recent: deque[dict[str, Any]] = deque(maxlen=max(1, args.logging_steps))
    history: list[dict[str, Any]] = []
    global_step = 0
    update_step = 0

    for step in range(1, args.max_steps + 1):
        global_step = step
        row = rows[(step - 1) % len(rows)]
        if step > 1 and (step - 1) % len(rows) == 0:
            random.shuffle(rows)
        db_path = sqlite_path(Path(args.sqlite_root), str(row["db_id"]))

        loss, metrics = grpo_step(model, tokenizer, row, db_path, args, device)
        (loss / args.gradient_accumulation_steps).backward()
        metrics.update({"global_step": global_step, "id": row.get("id"), "db_id": row.get("db_id")})
        recent.append(metrics)
        history.append({key: value for key, value in metrics.items() if key != "rollouts"})

        if step <= args.rollout_log_steps or step % max(1, args.rollout_log_steps) == 0:
            append_jsonl(
                rollout_log_path,
                {
                    "global_step": global_step,
                    "id": row.get("id"),
                    "db_id": row.get("db_id"),
                    "question": row.get("question"),
                    "gold_sql": gold_sql(row),
                    "best_sql": metrics.get("best_sql"),
                    "best_reward": metrics.get("best_reward"),
                    "best_exec_match": metrics.get("best_exec_match"),
                    "rollouts": metrics.get("rollouts", []),
                },
            )

        if step % args.gradient_accumulation_steps == 0 or step == args.max_steps:
            update_step += 1
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * lr_scale(update_step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if update_step == 1 or update_step % args.logging_steps == 0:
                averaged = mean_numeric(list(recent))
                averaged.update({"global_step": global_step, "update_step": update_step, "lr": optimizer.param_groups[0]["lr"]})
                print(json.dumps(averaged, ensure_ascii=False), flush=True)

    summary = {
        **config,
        "global_steps": global_step,
        "update_steps": update_step,
        "train_metrics": mean_numeric(history),
        "baseline_current_mainline_selected": "428/500",
        "baseline_current_value_linking_oracle": "451/500",
        "success_condition": "eval500 oracle > 451 and reranked selected > 428",
    }
    write_json(output_dir / "train_results.json", summary)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(json.dumps({"saved_checkpoint": args.output_dir, **summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
