#!/usr/bin/env python3
"""CLI demo for a tool-calling BI SQL agent."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rerank_sql_candidates import (  # noqa: E402
    evaluate_candidates,
    mentioned_columns,
    mentioned_tables,
    read_jsonl,
    read_only_sql,
    resolve_sqlite_path,
    schema_columns,
    schema_table_names,
    schema_issues,
)
from scripts.run_sql_candidate_predictions import (  # noqa: E402
    build_prompt,
    generate_candidates,
    has_output_pollution,
)


def load_schema_index(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_row(rows: list[dict[str, Any]], row_id: str | None) -> dict[str, Any]:
    if row_id:
        for row in rows:
            if row.get("id") == row_id:
                return row
        raise KeyError(f"row id not found: {row_id}")
    if not rows:
        raise ValueError("input file is empty")
    return rows[0]


def execute_sql_with_columns(sqlite_path: str, sql: str, timeout_sec: float) -> tuple[bool, list[str], list[list[Any]] | None, str | None]:
    uri = f"file:{sqlite_path}?mode=ro"
    conn = None
    try:
        deadline = time.monotonic() + timeout_sec
        conn = sqlite3.connect(uri, uri=True, timeout=timeout_sec)
        conn.execute("PRAGMA query_only = ON")
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = [list(row) for row in cursor.fetchall()]
        return True, columns, rows, None
    except Exception as exc:  # noqa: BLE001
        return False, [], None, f"{type(exc).__name__}: {exc}"
    finally:
        if conn is not None:
            conn.close()


def summarize_result(question: str, columns: list[str], rows: list[list[Any]] | None) -> str:
    if rows is None:
        return "No result because the SQL did not execute."
    if not rows:
        return "The query returned no rows."
    if len(rows) == 1 and len(columns) == 1:
        return f"The answer is {rows[0][0]}."
    if len(rows) == 1:
        parts = [f"{col}={value}" for col, value in zip(columns, rows[0], strict=False)]
        return "The answer is " + ", ".join(parts) + "."
    preview_rows = rows[:5]
    preview = "; ".join(
        ", ".join(str(value) for value in row)
        for row in preview_rows
    )
    return f"The query returned {len(rows)} rows. Sample: {preview}."


def make_schema_tool_output(question: str, row: dict[str, Any], schema_info: dict[str, Any], sqlite_path: str | None) -> dict[str, Any]:
    tables = schema_table_names(schema_info)
    columns = schema_columns(schema_info)
    matched_tables = sorted(mentioned_tables(question, schema_info))
    matched_columns = sorted(mentioned_columns(question, schema_info))
    table_summaries = [
        {
            "table": table,
            "columns": sorted(columns.get(table.lower(), set())),
        }
        for table in tables
    ]
    return {
        "db_id": row.get("db_id"),
        "sqlite_path": sqlite_path,
        "matched_tables": matched_tables,
        "matched_columns": matched_columns,
        "tables": table_summaries,
        "foreign_keys": schema_info.get("foreign_keys", []),
        "schema_version": schema_info.get("schema_version"),
    }


def format_block(title: str, payload: Any) -> str:
    return f"## {title}\n{json.dumps(payload, ensure_ascii=False, indent=2)}"


def load_generation_model(model_path: str, adapter_path: str | None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def build_online_source_row(args: argparse.Namespace, schema_info: dict[str, Any]) -> dict[str, Any]:
    row_id = args.row_id or f"online_{args.db_id}"
    row = {
        "id": row_id,
        "source": "online_demo",
        "db_id": args.db_id,
        "question": args.question,
        "schema": schema_info.get("schema_v2") or schema_info.get("schema") or "",
        "gold_sql": None,
        "gold_result": None,
        "expected_tools": ["search_schema", "validate_sql", "run_sql"],
        "expected_args": {"db_id": args.db_id},
    }
    row["prompt"] = build_prompt(row)
    return row


def attach_online_candidates(source: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer = load_generation_model(args.model_path, args.adapter_path)
    generated = generate_candidates(
        model=model,
        tokenizer=tokenizer,
        prompt=source["prompt"],
        max_new_tokens=args.max_new_tokens,
        sample_count=max(0, args.num_candidates - 1),
        temperature=args.temperature,
        top_p=args.top_p,
    )

    candidates = []
    for rank, (method, raw, prediction) in enumerate(generated[: args.num_candidates], start=1):
        candidates.append(
            {
                "rank": rank,
                "method": method,
                "raw_prediction": raw,
                "prediction": prediction,
                "raw_output_polluted": has_output_pollution(raw),
            }
        )

    source = dict(source)
    source["candidates"] = candidates
    source["prediction"] = candidates[0]["prediction"] if candidates else ""
    source["model_path"] = args.model_path
    source["adapter_path"] = args.adapter_path
    source["prompt_template"] = "online_completion_sql_candidates"
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-id", default=None)
    parser.add_argument("--db-id", default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--sql", default=None)
    parser.add_argument("--generate-candidates", action="store_true")
    parser.add_argument("--model-path", default=str(ROOT / "hf_cache" / "models" / "Qwen3-4B-Base"))
    parser.add_argument("--adapter-path", default=str(ROOT / "checkpoints" / "sft" / "sql_sft_v3_qwen3_4b"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--candidates-file", default=str(ROOT / "logs" / "rerank_sft_v3_schema_v2_candidates_eval500.jsonl"))
    parser.add_argument("--schema-index", default=str(ROOT / "data" / "processed" / "spider_schema_v2.json"))
    parser.add_argument("--sqlite-root", default=str(ROOT / "data" / "raw" / "spider" / "database"))
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    parser.add_argument("--trace-output", default=None)
    parser.add_argument("--list-examples", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    schema_index = load_schema_index(Path(args.schema_index))

    if args.list_examples:
        rows = read_jsonl(Path(args.candidates_file))
        if args.limit is not None:
            rows = rows[: args.limit]
        for row in rows:
            print(f'{row.get("id")} | {row.get("db_id")} | {row.get("question")}')
        return

    if args.generate_candidates:
        if not args.db_id or not args.question:
            raise ValueError("--generate-candidates requires --db-id and --question")
        schema_info = schema_index.get(args.db_id)
        if not schema_info:
            raise KeyError(f"schema not found for db_id={args.db_id}")
        source = build_online_source_row(args, schema_info)
        source = attach_online_candidates(source, args)
        db_id = source["db_id"]
        sqlite_path = resolve_sqlite_path(schema_info, args.sqlite_root)
        evaluated, selected = evaluate_candidates(
            source,
            schema_index,
            args.sqlite_root,
            args.timeout_sec,
            enable_join_graph=False,
        )
        selected_sql = selected.get("pred_sql") if selected else ""
        selected_meta = {
            "method": selected.get("method") if selected else None,
            "rank": selected.get("rank") if selected else None,
            "rerank_score": selected.get("rerank_score") if selected else None,
            "exec_match": bool(selected and selected.get("exec_match")),
            "candidate_count": len(source.get("candidates", [])),
            "generation_mode": "online",
        }
        candidates_trace = evaluated
    elif args.sql:
        source_rows = read_jsonl(Path(args.candidates_file))
        source = find_row(source_rows, args.row_id)
        db_id = source.get("db_id")
        if not db_id:
            raise ValueError("db_id missing in source row")
        schema_info = schema_index.get(db_id)
        if not schema_info:
            raise KeyError(f"schema not found for db_id={db_id}")
        sqlite_path = resolve_sqlite_path(schema_info, args.sqlite_root)
        selected_sql = args.sql.strip()
        selected_meta = {"method": "manual_sql", "rank": None, "rerank_score": None}
        candidates_trace = None
    else:
        source_rows = read_jsonl(Path(args.candidates_file))
        source = find_row(source_rows, args.row_id)
        db_id = source.get("db_id")
        if not db_id:
            raise ValueError("db_id missing in source row")
        schema_info = schema_index.get(db_id)
        if not schema_info:
            raise KeyError(f"schema not found for db_id={db_id}")
        sqlite_path = resolve_sqlite_path(schema_info, args.sqlite_root)
        evaluated, selected = evaluate_candidates(
            source,
            schema_index,
            args.sqlite_root,
            args.timeout_sec,
            enable_join_graph=False,
        )
        selected_sql = selected.get("pred_sql") if selected else ""
        selected_meta = {
            "method": selected.get("method") if selected else None,
            "rank": selected.get("rank") if selected else None,
            "rerank_score": selected.get("rerank_score") if selected else None,
            "exec_match": bool(selected and selected.get("exec_match")),
        }
        candidates_trace = evaluated

    question = source.get("question", "")
    search_schema_obs = make_schema_tool_output(question, source, schema_info, sqlite_path)
    validate_obs = {
        "sql": selected_sql,
        "read_only": read_only_sql(selected_sql),
        "schema_issues": schema_issues(selected_sql, schema_info),
        "sqlite_available": bool(sqlite_path and Path(sqlite_path).exists()),
    }
    validate_obs["valid"] = bool(validate_obs["read_only"] and not validate_obs["schema_issues"] and validate_obs["sqlite_available"])

    if not validate_obs["valid"]:
        run_obs = {
            "executed": False,
            "error": "validation_failed",
        }
        final_answer = "The SQL did not pass validation, so the agent stopped before running it."
    else:
        executed, columns, rows, error = execute_sql_with_columns(sqlite_path, selected_sql, args.timeout_sec)
        run_obs = {
            "executed": executed,
            "columns": columns,
            "rows": rows[:5] if rows else [],
            "row_count": len(rows) if rows is not None else None,
            "error": error,
        }
        final_answer = summarize_result(question, columns, rows)

    trace = {
        "id": source.get("id"),
        "question": question,
        "db_id": db_id,
        "source_row": source,
        "search_schema": search_schema_obs,
        "selected_sql": selected_sql,
        "selected_meta": selected_meta,
        "validate_sql": validate_obs,
        "run_sql": run_obs,
        "final_answer": final_answer,
        "candidates_trace": candidates_trace,
    }

    print(format_block("search_schema", search_schema_obs))
    print(format_block("model_sql", {"sql": selected_sql, **selected_meta}))
    print(format_block("validate_sql", validate_obs))
    print(format_block("run_sql", run_obs))
    print(format_block("final_answer", {"answer": final_answer}))

    if args.trace_output:
        trace_path = Path(args.trace_output)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
