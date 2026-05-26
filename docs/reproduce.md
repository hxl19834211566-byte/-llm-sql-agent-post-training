# 复现说明

这个文档只保留最终主线的复现路径。完整实验数据、模型权重、候选 SQL 文件和 SQLite 数据库需要在本地准备，不随仓库提交。

## 最终结果

```text
SFT v3 + schema v2 + value-linking v1 + n20 candidates + rerank v15
= 450/500
= 90.0% SQLite 执行准确率
```

候选集 oracle 上限：

```text
451/500 = 90.2%
```

## 本地资产

需要准备：

```text
hf_cache/models/Qwen3-4B-Base
checkpoints/sft/sql_sft_v3_qwen3_4b
data/raw/spider/tables.json
data/raw/spider/database/<db_id>/<db_id>.sqlite
```

如果要从头构建 schema v2：

```bash
python scripts/build_spider_schema_v2.py \
  --tables-json data/raw/spider/tables.json \
  --database-dir data/raw/spider/database \
  --output data/processed/spider_schema_v2.json
```

## 生成 value-linking eval

```bash
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

## 生成 n20 candidates

这一步需要 GPU 和 SFT v3 adapter。

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

## 运行最终 rerank

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

注意：最终报告结果不要添加 `--enable-join-graph`。

预期核心 summary：

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
