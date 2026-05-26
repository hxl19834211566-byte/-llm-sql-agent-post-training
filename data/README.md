# 数据工作区

这个目录用于本地存放 benchmark 数据、训练 / 评估 JSONL 文件，以及派生的 schema index。

仓库不重新分发原始 benchmark 数据库、训练数据、预测结果或大型 JSONL 文件。需要复现时，在本地重新生成或手动放置这些文件。

完整复现时的本地目录结构：

```text
data/raw/spider/tables.json
data/raw/spider/database/<db_id>/<db_id>.sqlite
data/eval/spider_schema_eval_500.jsonl
data/eval/spider_schema_v2_eval_500.jsonl
data/eval/spider_schema_v2_value_linking_eval_500.jsonl
data/processed/spider_schema_v2.json
```

常用构建命令：

```bash
python scripts/build_spider_schema_v2.py \
  --tables-json data/raw/spider/tables.json \
  --database-dir data/raw/spider/database \
  --output data/processed/spider_schema_v2.json

python scripts/upgrade_jsonl_schema_v2.py \
  --input data/eval/spider_schema_eval_500.jsonl \
  --schema-index data/processed/spider_schema_v2.json \
  --output data/eval/spider_schema_v2_eval_500.jsonl \
  --summary-output logs/spider_schema_v2_eval_500_summary.json

python scripts/upgrade_jsonl_value_linking.py \
  --input data/eval/spider_schema_v2_eval_500.jsonl \
  --sqlite-root data/raw/spider/database \
  --output data/eval/spider_schema_v2_value_linking_eval_500.jsonl \
  --summary-output logs/spider_schema_v2_value_linking_eval_500_summary.json
```
