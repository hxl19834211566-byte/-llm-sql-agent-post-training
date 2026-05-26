# Data Workspace

This directory is a local workspace for benchmark data, generated train/eval JSONL files, and derived schema indexes.

Raw benchmark databases, generated training data, prediction dumps, and large derived JSONL artifacts are not redistributed. Recreate or place them locally as needed.

Expected local layout for full reproduction:

```text
data/raw/spider/tables.json
data/raw/spider/database/<db_id>/<db_id>.sqlite
data/eval/day2_spider_schema_eval_500.jsonl
data/eval/day2_spider_schema_v2_eval_500.jsonl
data/eval/day2_spider_schema_v2_value_linking_eval_500.jsonl
data/processed/spider_schema_v2.json
```

Useful builders:

```bash
python scripts/build_spider_schema_v2.py \
  --tables-json data/raw/spider/tables.json \
  --database-dir data/raw/spider/database \
  --output data/processed/spider_schema_v2.json

python scripts/upgrade_jsonl_schema_v2.py \
  --input data/eval/day2_spider_schema_eval_500.jsonl \
  --schema-index data/processed/spider_schema_v2.json \
  --output data/eval/day2_spider_schema_v2_eval_500.jsonl \
  --summary-output logs/day2_spider_schema_v2_eval_500_summary.json

python scripts/upgrade_jsonl_value_linking.py \
  --input data/eval/day2_spider_schema_v2_eval_500.jsonl \
  --sqlite-root data/raw/spider/database \
  --output data/eval/day2_spider_schema_v2_value_linking_eval_500.jsonl \
  --summary-output logs/day2_spider_schema_v2_value_linking_eval_500_summary.json
```
