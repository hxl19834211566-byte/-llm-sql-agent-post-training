# Value Linking v1 Runbook

Date: 2026-05-12

## Goal

Test a prompt-only generation-time value linking route:

```text
SFT v3 + schema v2 + value-linking prompt candidates + rerank v14
```

This is inference-time prompt augmentation. It does not train a model and does not update SFT v3 weights.

Current baseline to beat:

```text
selected = 407/500 = 81.4%
oracle   = 409/500 = 81.8%
```

Success rule:

```text
selected_exec_match > 407/500
oracle_exec_match > 409/500
```

`selected` decides adoption. `oracle` decides whether the value-linking generation route has potential.

## Method

Script:

```text
scripts/upgrade_jsonl_value_linking.py
```

It performs:

```text
question
-> rule-based mention extraction
-> SQLite text-column retrieval inside the current db_id
-> same-row ID / abbreviation / name hint extraction
-> append compact value hints to schema text
-> SFT v3 generates candidates from the augmented prompt
-> rerank v14 selects final SQL
```

Example hint:

```text
-- Value linking hints matched from the question:
-- AIRLINES.Airline = "JetBlue Airways"; same row: uid=...; Airline="JetBlue Airways"; Abbreviation=...; Country=...
```

The hint is inserted into `row["schema"]`, so existing generation scripts can be reused unchanged.

## Requirements

Needs a machine with:

```text
Qwen3-4B-Base
checkpoints/sft/sql_sft_v3_qwen3_4b
data/raw/spider/database
data/eval/day2_spider_schema_v2_eval_500.jsonl
data/processed/spider_schema_v2.json
```

Local note as of 2026-05-12:

```text
Local Windows has SFT v3 LoRA and schema_v2.json.
Local Windows does not have Qwen3-4B-Base or Spider SQLite database.
```

## Step 1. Build Value-Linking Eval

Run on the server or 4090 machine after Spider SQLite DB exists:

```bash
python scripts/upgrade_jsonl_value_linking.py \
  --input data/eval/day2_spider_schema_v2_eval_500.jsonl \
  --sqlite-root data/raw/spider/database \
  --output data/eval/day2_spider_schema_v2_value_linking_eval_500.jsonl \
  --summary-output logs/day2_spider_schema_v2_value_linking_eval_500_summary.json \
  --max-hints 8 \
  --max-mentions 24 \
  --min-score 82 \
  --max-rows-per-query 50
```

Check summary:

```bash
cat logs/day2_spider_schema_v2_value_linking_eval_500_summary.json
```

Expected:

```text
rows = 500
rows_with_hints > 0
missing_sqlite = 0
```

If `missing_sqlite > 0`, fix SQLite root before generating candidates.

## Step 2. Generate Candidates

First run candidate count 10:

```bash
python scripts/run_sql_candidate_predictions.py \
  --model-path hf_cache/models/Qwen3-4B-Base \
  --adapter-path checkpoints/sft/sql_sft_v3_qwen3_4b \
  --eval-file data/eval/day2_spider_schema_v2_value_linking_eval_500.jsonl \
  --output-file logs/rerank_sft_v3_schema_v2_value_linking_candidates_eval500_n10.jsonl \
  --max-new-tokens 128 \
  --num-candidates 10 \
  --temperature 0.7 \
  --top-p 0.9
```

If runtime is acceptable and n10 improves oracle, run n20:

```bash
python scripts/run_sql_candidate_predictions.py \
  --model-path hf_cache/models/Qwen3-4B-Base \
  --adapter-path checkpoints/sft/sql_sft_v3_qwen3_4b \
  --eval-file data/eval/day2_spider_schema_v2_value_linking_eval_500.jsonl \
  --output-file logs/rerank_sft_v3_schema_v2_value_linking_candidates_eval500_n20.jsonl \
  --max-new-tokens 128 \
  --num-candidates 20 \
  --temperature 0.7 \
  --top-p 0.9
```

## Step 3. Rerank with v14

For n10:

```bash
python scripts/rerank_sql_candidates.py \
  --candidates logs/rerank_sft_v3_schema_v2_value_linking_candidates_eval500_n10.jsonl \
  --schema-index data/processed/spider_schema_v2.json \
  --sqlite-root data/raw/spider/database \
  --selected-output logs/rerank_sft_v3_schema_v2_value_linking_selected_eval500_n10_v14.jsonl \
  --analysis-output logs/rerank_sft_v3_schema_v2_value_linking_analysis_eval500_n10_v14.jsonl \
  --summary-output logs/rerank_sft_v3_schema_v2_value_linking_summary_eval500_n10_v14.json \
  --timeout-sec 5
```

For n20:

```bash
python scripts/rerank_sql_candidates.py \
  --candidates logs/rerank_sft_v3_schema_v2_value_linking_candidates_eval500_n20.jsonl \
  --schema-index data/processed/spider_schema_v2.json \
  --sqlite-root data/raw/spider/database \
  --selected-output logs/rerank_sft_v3_schema_v2_value_linking_selected_eval500_n20_v14.jsonl \
  --analysis-output logs/rerank_sft_v3_schema_v2_value_linking_analysis_eval500_n20_v14.jsonl \
  --summary-output logs/rerank_sft_v3_schema_v2_value_linking_summary_eval500_n20_v14.json \
  --timeout-sec 5
```

Do not use `--enable-join-graph`.

## Decision

Adopt only if:

```text
selected_exec_match > 407
```

Interpretation:

| Result | Decision |
| --- | --- |
| `selected > 407` and `oracle > 409` | adopt candidate route, update mainline |
| `oracle > 409` but `selected <= 407` | generation route works; improve selection next |
| `selected > 407` but `oracle <= 409` | verify with repeat run; likely selection/sampling effect |
| `selected <= 407` and `oracle <= 409` | do not adopt |

## Notes

This v1 is rule-based:

```text
no value-linking model
no embedding index
no model training
```

If v1 improves oracle, a later v2 can add embedding retrieval or targeted execution-guided repair.
