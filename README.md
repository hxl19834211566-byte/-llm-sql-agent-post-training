# Text-to-SQL Agent: SFT + Value Linking + Rerank

BI / Text-to-SQL pipeline with executable SQL evaluation. The system uses a Qwen3-4B-Base LoRA SFT checkpoint, schema-aware prompting, value-linking hints, multi-candidate generation, and a deterministic rerank/repair stage.

Main result:

```text
SFT v3 + schema v2 + value-linking v1 + n20 candidates + rerank v15
= 450 / 500
= 90.0% SQLite execution accuracy
```

The candidate oracle upper bound is `451 / 500 = 90.2%`. That number is used only for analysis; the selected result is `90.0%`.

## Contents

- Core SFT, inference, and evaluation scripts.
- Optional DPO / OPD / ORPO / GRPO side-route scripts.
- SQL execution evaluation on Spider-style SQLite databases.
- Schema v2 and value-linking prompt builders.
- Multi-candidate SQL generation.
- Deterministic SQL candidate rerank/repair.
- CLI demo for a tool-calling SQL agent flow.
- Final reports and result summaries.

Large local artifacts are not tracked:

- Base models and Hugging Face caches.
- LoRA adapter weights.
- Spider SQLite database files.
- Generated JSONL predictions, candidate dumps, and raw train/eval data.
- Local notes and generated artifacts.

## Repository Layout

```text
demo/        Tool-calling SQL agent demo
docs/        Mainline reports and runbooks
eval/        SQLite execution evaluator
results/     Result summaries
scripts/     Core data prep, training, generation, value-linking, rerank
data/        Local-only data workspace, documented but mostly ignored
checkpoints/ Local-only adapter workspace, documented but weights ignored
```

## Final Pipeline

```text
Qwen3-4B-Base
-> SFT v3 LoRA
-> schema v2 prompt
-> value-linking v1 prompt hints
-> 20 SQL candidates per question
-> rerank / repair v15
-> SQLite execution evaluation
```

SFT v3 fixed most output-format failures. Later errors were mostly schema linking, value grounding, joins, aliases, and aggregation semantics, so the final gains came from candidate generation and reranking rather than another training pass.

## Key Results

| System | Execution accuracy |
| --- | ---: |
| Qwen3-4B-Base baseline | 313/500 = 62.6% |
| SFT v3 greedy | 356/500 = 71.2% |
| SFT v3 + schema v2 prompt-only | 361/500 = 72.2% |
| SFT v3 + schema v2 + rerank v14 | 407/500 = 81.4% |
| SFT v3 + schema v2 + value-linking n20 + rerank v14 | 428/500 = 85.6% |
| SFT v3 + schema v2 + value-linking n20 + rerank v15 | 450/500 = 90.0% |

See [results/leaderboard.md](results/leaderboard.md) and [results/final_mainline_summary.json](results/final_mainline_summary.json).

## Setup

Environment used for the recorded runs: Ubuntu 22.04, Python 3.10, CUDA 12.8, A800 80GB.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place external assets locally:

```text
hf_cache/models/Qwen3-4B-Base
checkpoints/sft/sql_sft_v3_qwen3_4b
data/raw/spider/tables.json
data/raw/spider/database/<db_id>/<db_id>.sqlite
```

These files are not committed because they are large or externally licensed.

## Reproduce The Final Rerank

If you have the saved candidate JSONL locally:

```bash
python scripts/rerank_sql_candidates.py \
  --candidates logs/rerank_sft_v3_schema_v2_value_linking_candidates_eval500_n20.jsonl \
  --schema-index data/processed/spider_schema_v2.json \
  --sqlite-root data/raw/spider/database \
  --selected-output logs/rerank_sft_v3_schema_v2_value_linking_selected_eval500_n20_v15_repro.jsonl \
  --analysis-output logs/rerank_sft_v3_schema_v2_value_linking_analysis_eval500_n20_v15_repro.jsonl \
  --summary-output logs/rerank_sft_v3_schema_v2_value_linking_summary_eval500_n20_v15_repro.json \
  --timeout-sec 5 \
  --score-version v15
```

Expected core summary:

```json
{
  "total": 500,
  "selected_exec_match": 450,
  "selected_execution_accuracy": 0.9,
  "oracle_exec_match": 451,
  "oracle_execution_accuracy": 0.902,
  "enable_join_graph": false,
  "score_version": "v15"
}
```

Do not pass `--enable-join-graph` for this result.

## Run The Demo

List examples from a local candidate file:

```bash
python demo/sql_agent_demo.py \
  --candidates-file logs/rerank_sft_v3_schema_v2_value_linking_candidates_eval500_n20.jsonl \
  --schema-index data/processed/spider_schema_v2.json \
  --list-examples \
  --limit 5
```

Run one offline tool-calling trace:

```bash
python demo/sql_agent_demo.py \
  --row-id day2_spider_schema_val_0001 \
  --candidates-file logs/rerank_sft_v3_schema_v2_value_linking_candidates_eval500_n20.jsonl \
  --schema-index data/processed/spider_schema_v2.json \
  --sqlite-root data/raw/spider/database \
  --trace-output logs/sql_agent_demo_trace.json
```

The demo prints `search_schema`, selected model SQL, `validate_sql`, `run_sql`, and a grounded final answer.

## Recommended Reading

- [docs/mainline_report.md](docs/mainline_report.md)
- [docs/final_pipeline_runbook.md](docs/final_pipeline_runbook.md)
- [docs/side_routes.md](docs/side_routes.md)
- [docs/value_linking_v1_report.md](docs/value_linking_v1_report.md)
- [docs/rerank_repair_report.md](docs/rerank_repair_report.md)
- [docs/tool_calling_demo_report.md](docs/tool_calling_demo_report.md)

## License

Before publishing, choose a license for the code. Model weights, datasets, and benchmark assets keep their upstream licenses and are not redistributed here.
