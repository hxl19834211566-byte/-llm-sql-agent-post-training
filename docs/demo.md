# Demo

`demo/sql_agent_demo.py` 是一个离线 SQL Agent demo，用同一套候选生成、校验、执行和返回结果流程。

流程：

```text
问题
-> search_schema
-> 选择 / 生成 SQL
-> validate_sql
-> run_sql
-> final_answer
```

## 列出候选样例

```bash
python demo/sql_agent_demo.py \
  --candidates-file logs/rerank_sft_v3_schema_v2_value_linking_candidates_eval500_n20.jsonl \
  --schema-index data/processed/spider_schema_v2.json \
  --list-examples \
  --limit 5
```

## 运行离线 trace

先用 `--list-examples` 找到本地候选文件里的样例 id，再传给 `--row-id`。

```bash
python demo/sql_agent_demo.py \
  --row-id spider_eval_0001 \
  --candidates-file logs/rerank_sft_v3_schema_v2_value_linking_candidates_eval500_n20.jsonl \
  --schema-index data/processed/spider_schema_v2.json \
  --sqlite-root data/raw/spider/database \
  --trace-output logs/sql_agent_demo_trace.json
```

输出内容包括：

```text
search_schema
model_sql
validate_sql
run_sql
final_answer
```

## 在线生成模式

如果本地有 base model 和 SFT v3 adapter，可以直接对新问题生成 candidates：

```bash
python demo/sql_agent_demo.py \
  --generate-candidates \
  --db-id concert_singer \
  --question "How many singers do we have?" \
  --model-path hf_cache/models/Qwen3-4B-Base \
  --adapter-path checkpoints/sft/sql_sft_v3_qwen3_4b \
  --num-candidates 5 \
  --schema-index data/processed/spider_schema_v2.json \
  --sqlite-root data/raw/spider/database
```
