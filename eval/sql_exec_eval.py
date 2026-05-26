#!/usr/bin/env python3
"""Execute predicted SQL and gold SQL on Spider SQLite databases."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def extract_sql(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    # Keep from the first SELECT/WITH if the model adds prose.
    match = re.search(r"\b(select|with)\b", text, flags=re.IGNORECASE)
    if match:
        text = text[match.start() :].strip()

    cutoff_markers = [
        "NdrFcShort",
        "<|im_end|>",
        "<|im_start|>",
        "\nTask:",
        "\nDatabase id:",
        "\nSchema:",
        "\nQuestion:",
        "\nuser\n",
        "\nassistant\n",
    ]
    cutoff_positions = [text.find(marker) for marker in cutoff_markers if marker in text]
    if cutoff_positions:
        text = text[: min(cutoff_positions)].strip()

    # Use the first SQL statement for Spider-style eval.
    if ";" in text:
        text = text.split(";", 1)[0].strip()
    return text


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
        conn.execute(f"PRAGMA query_only = ON")
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        return True, [list(row) for row in rows], None
    except Exception as exc:  # noqa: BLE001
        return False, None, f"{type(exc).__name__}: {exc}"
    finally:
        if conn is not None:
            conn.close()


def evaluate_row(row: dict[str, Any], schema_index: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    db_id = row.get("db_id") or row.get("expected_args", {}).get("db_id")
    pred_sql = extract_sql(row.get("prediction", ""))
    gold_sql = row.get("gold_sql") or ""
    db_info = schema_index.get(db_id)

    result: dict[str, Any] = {
        "id": row.get("id"),
        "source": row.get("source"),
        "db_id": db_id,
        "question": row.get("question"),
        "prediction": row.get("prediction"),
        "pred_sql": pred_sql,
        "gold_sql": gold_sql,
        "sqlite_path": db_info.get("sqlite_path") if db_info else None,
        "schema_available": bool(db_info),
        "pred_read_only_shape": read_only_sql(pred_sql),
    }

    if not db_info or not Path(db_info["sqlite_path"]).exists():
        result.update(
            {
                "gold_exec_success": False,
                "pred_exec_success": False,
                "exec_match": False,
                "error": "missing_sqlite",
            }
        )
        return result

    gold_ok, gold_rows, gold_error = execute_sql(db_info["sqlite_path"], gold_sql, timeout_sec)
    pred_ok, pred_rows, pred_error = execute_sql(db_info["sqlite_path"], pred_sql, timeout_sec)
    ordered = has_order_by(gold_sql)
    gold_norm = normalize_rows([tuple(row) for row in (gold_rows or [])], ordered) if gold_ok else None
    pred_norm = normalize_rows([tuple(row) for row in (pred_rows or [])], ordered) if pred_ok else None

    result.update(
        {
            "gold_exec_success": gold_ok,
            "pred_exec_success": pred_ok,
            "gold_error": gold_error,
            "pred_error": pred_error,
            "ordered_compare": ordered,
            "gold_result": gold_norm,
            "pred_result": pred_norm,
            "exec_match": bool(gold_ok and pred_ok and gold_norm == pred_norm),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--schema-index", default="/root/project/data/processed/spider_schema.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    args = parser.parse_args()

    rows = [row for row in read_jsonl(Path(args.predictions)) if row.get("source") == "xlangai/spider"]
    schema_index = json.loads(Path(args.schema_index).read_text(encoding="utf-8"))
    evaluated = [evaluate_row(row, schema_index, args.timeout_sec) for row in rows]

    total = len(evaluated)
    summary = {
        "predictions": args.predictions,
        "total": total,
        "schema_available": sum(row["schema_available"] for row in evaluated),
        "pred_read_only_shape": sum(row["pred_read_only_shape"] for row in evaluated),
        "gold_exec_success": sum(row["gold_exec_success"] for row in evaluated),
        "pred_exec_success": sum(row["pred_exec_success"] for row in evaluated),
        "exec_match": sum(row["exec_match"] for row in evaluated),
        "execution_accuracy": (sum(row["exec_match"] for row in evaluated) / total) if total else 0.0,
        "pred_exec_success_rate": (sum(row["pred_exec_success"] for row in evaluated) / total) if total else 0.0,
    }

    write_jsonl(Path(args.output), evaluated)
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
