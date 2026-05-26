# Tool-Calling Demo Report

Date: 2026-05-11

## Goal

把当前 SQL 主线包装成一个最小可演示的 BI SQL Agent：

```text
search_schema -> validate_sql -> run_sql -> final answer
```

这里的 Agent 不是新训练的模型，而是把已经固定的主线能力接到工具闭环上。

## Demo Script

脚本：

```text
demo/sql_agent_demo.py
```

输入：

- `row_id`
- `candidates-file`
- `schema-index`
- `sqlite-root`

输出：

- `search_schema`
- `model_sql`
- `validate_sql`
- `run_sql`
- `final_answer`

## Demo Data

默认使用已有的 schema v2 candidates：

```text
logs/rerank_sft_v3_schema_v2_candidates_eval500.jsonl
```

默认 schema index：

```text
data/processed/spider_schema_v2.json
```

默认 SQLite 根目录：

```text
data/raw/spider/database
```

如果在服务器上运行，这些路径都指向同一套 Spider DB 资产。

## Tool Flow

### 1. `search_schema`

工具输入：

```json
{"db_id": "...", "question": "..."}
```

工具输出：

- `matched_tables`
- `matched_columns`
- `tables`
- `foreign_keys`
- `sqlite_path`

这一步负责把问题绑定到当前数据库 schema，不直接生成 SQL。

### 2. `model_sql`

这里不是新训练的生成器，而是复用当前已固定的候选选择逻辑：

```text
SFT v3 candidates -> rerank/repair -> selected SQL
```

这保持了 demo 和主结果的一致性。

### 3. `validate_sql`

检查：

- 是否只读
- 是否存在 schema 引用问题
- 是否有可用 SQLite 资产

### 4. `run_sql`

只在验证通过后执行 SQL，返回：

- 列名
- 前几行结果
- 总行数

### 5. `final_answer`

不再猜答案，直接根据 SQL 结果形成简短自然语言输出。

## Why This Is the Right Next Step

当前主结果已经固定：

```text
SFT v3 + schema v2 candidates + rerank v12/v13 = 404/500 = 80.8%
```

继续扩大训练的收益已经很小。把能力封装成工具型 Agent，比继续追逐小幅分数变化更符合当前主线目标。

## Expected Demo Command

在服务器上可直接跑：

```bash
cd /root/project
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py310
python demo/sql_agent_demo.py \
  --row-id day2_spider_schema_val_0001 \
  --candidates-file logs/rerank_sft_v3_schema_v2_candidates_eval500.jsonl \
  --schema-index data/processed/spider_schema_v2.json \
  --sqlite-root data/raw/spider/database \
  --trace-output logs/sql_agent_demo_day2_0001_trace.json
```

## Reported Example

推荐演示样例：

- `day2_spider_schema_val_0001`
- `day2_spider_schema_val_0003`
- `day2_spider_schema_val_0011`

这些样例的工具流简单，适合展示 `search_schema`、`validate_sql`、`run_sql` 的闭环。

## Current Conclusion

现在项目主线已经不是再训练，而是交付：

```text
SQL 主线结果 + 工具调用 demo + 最终报告
```

这样更完整，也更符合 BI SQL Agent 的故事线。
