# Mainline Report

## Result

Final selected pipeline:

```text
Qwen3-4B-Base
-> SFT v3 LoRA
-> schema v2 prompt
-> value-linking v1 hints
-> n20 SQL candidates
-> rerank / repair v15
```

Evaluation:

```text
eval set: Spider-style eval500
metric: SQLite execution accuracy
selected: 450/500 = 90.0%
oracle:   451/500 = 90.2%
```

The selected score is the deployable result. The oracle score is only the upper bound of the saved n20 candidate set.

## Why This Mainline

SFT v3 fixed the output boundary: the model reliably returns one read-only SQL statement. After that point, the main errors were schema linking, value grounding, joins, aliases, and projection semantics.

The final gains came from inference-time search:

1. `schema v2` improves table and column context.
2. `value-linking v1` retrieves database values and ID hints from SQLite.
3. `n20` candidate generation increases the chance that the correct SQL appears.
4. `rerank / repair v15` selects the best executable candidate with schema and question-aware rules.

## Progression

| System | Correct | Accuracy |
| --- | ---: | ---: |
| Qwen3-4B-Base baseline | 313/500 | 62.6% |
| SFT v3 greedy | 356/500 | 71.2% |
| SFT v3 + schema v2 prompt-only | 361/500 | 72.2% |
| SFT v3 + schema v2 + rerank v14 | 407/500 | 81.4% |
| SFT v3 + schema v2 + value-linking n10 + rerank v14 | 415/500 | 83.0% |
| SFT v3 + schema v2 + value-linking n20 + rerank v14 | 428/500 | 85.6% |
| SFT v3 + schema v2 + value-linking n20 + rerank v15 | 450/500 | 90.0% |

## Core Files

```text
scripts/build_spider_schema_v2.py
scripts/upgrade_jsonl_schema_v2.py
scripts/upgrade_jsonl_value_linking.py
scripts/run_sql_candidate_predictions.py
scripts/rerank_sql_candidates.py
eval/sql_exec_eval.py
demo/sql_agent_demo.py
```

Related reports:

```text
docs/final_pipeline_runbook.md
docs/value_linking_v1_report.md
docs/rerank_repair_report.md
docs/tool_calling_demo_report.md
results/final_mainline_summary.json
```
