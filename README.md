# Text-to-SQL 智能体：SFT、值链接与重排序

这个仓库整理了一套 Text-to-SQL / BI SQL Agent 流程。输入是自然语言问题、数据库 schema 和 `db_id`，输出一条可执行 SQLite SQL，并用执行结果判断是否正确。

当前采用的配置是 Qwen3-4B-Base + SFT v3 LoRA，再配合 schema v2 prompt、value-linking、多候选生成和 SQL rerank / repair。在固定 eval500 上的结果：

```text
SFT v3 + schema v2 + value-linking v1 + n20 candidates + rerank v15
= 450 / 500
= 90.0% SQLite 执行准确率
```

同一批 n20 候选里，最多可命中的数量是：

```text
451 / 500 = 90.2%
```

也就是说，当前选择器选出的结果是 `450/500`，候选本身还剩 1 条没有被排到第一。

## 仓库内容

- SFT 训练、推理和 SQLite 执行评估脚本。
- DPO / OPD / ORPO / GRPO 可选副线实验脚本。
- schema v2 prompt 构建与值链接提示增强。
- 多候选 SQL 生成与候选重排序 / 修复。
- 工具调用式 SQL Agent CLI demo。
- 整理后的结果摘要和复现说明。

这些内容需要放在本地，不放进仓库：

- Hugging Face base model cache。
- LoRA adapter 权重。
- Spider SQLite database。
- 生成的 JSONL 数据、候选 SQL、预测结果和训练 / 评估中间文件。
- 本地笔记和临时产物。

## 目录结构

```text
demo/        工具调用式 SQL Agent demo
docs/        复现说明、实验记录和 demo 用法
eval/        SQLite 执行评估器
results/     结果摘要
scripts/     数据处理、训练、生成、值链接、重排序脚本
data/        本地数据工作区，大部分内容被 gitignore
checkpoints/ 本地 checkpoint 工作区，权重被 gitignore
```

## 当前流程

```text
Qwen3-4B-Base
-> SFT v3 LoRA
-> schema v2 prompt
-> value-linking v1 prompt hints
-> 每题生成 20 个 SQL candidates
-> rerank / repair v15
-> SQLite execution evaluation
```

SFT v3 主要解决输出边界问题，让模型稳定输出一条只读 SQL。后面的错误更多来自 schema linking、value grounding、JOIN、别名、聚合和返回列语义，因此提升主要来自候选生成和重排序。

## 结果

| 方案 | 执行准确率 |
| --- | ---: |
| Qwen3-4B-Base baseline | 313/500 = 62.6% |
| SFT v3 greedy | 356/500 = 71.2% |
| SFT v3 + schema v2 prompt-only | 361/500 = 72.2% |
| SFT v3 + schema v2 + rerank v14 | 407/500 = 81.4% |
| SFT v3 + schema v2 + value-linking n20 + rerank v14 | 428/500 = 85.6% |
| SFT v3 + schema v2 + value-linking n20 + rerank v15 | 450/500 = 90.0% |

更多结果：

- [results/leaderboard.md](results/leaderboard.md)
- [results/final_mainline_summary.json](results/final_mainline_summary.json)

## 环境

记录实验使用的环境：

```text
Ubuntu 22.04
Python 3.10
CUDA 12.8
A800 80GB
```

安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

需要本地放置的外部资产：

```text
hf_cache/models/Qwen3-4B-Base
checkpoints/sft/sql_sft_v3_qwen3_4b
data/raw/spider/tables.json
data/raw/spider/database/<db_id>/<db_id>.sqlite
```

这些文件体积较大或受上游许可证约束，因此不随仓库发布。

## 复现 rerank 结果

如果本地已有保存好的 n20 candidates：

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

预期 summary：

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

这组结果使用默认 rerank 配置，不开启 `--enable-join-graph`。

## 运行 Demo

列出本地候选文件中的样例：

```bash
python demo/sql_agent_demo.py \
  --candidates-file logs/rerank_sft_v3_schema_v2_value_linking_candidates_eval500_n20.jsonl \
  --schema-index data/processed/spider_schema_v2.json \
  --list-examples \
  --limit 5
```

运行一个离线工具调用 trace：

```bash
python demo/sql_agent_demo.py \
  --row-id spider_eval_0001 \
  --candidates-file logs/rerank_sft_v3_schema_v2_value_linking_candidates_eval500_n20.jsonl \
  --schema-index data/processed/spider_schema_v2.json \
  --sqlite-root data/raw/spider/database \
  --trace-output logs/sql_agent_demo_trace.json
```

`--row-id` 可以换成 `--list-examples` 输出的样例 id。demo 会输出：

```text
search_schema
model_sql
validate_sql
run_sql
final_answer
```

## 文档入口

- [docs/reproduce.md](docs/reproduce.md)：复现 90.0% 结果。
- [docs/experiments.md](docs/experiments.md)：主要流程和 DPO / OPD / ORPO / GRPO 副线实验。
- [docs/demo.md](docs/demo.md)：工具调用式 SQL 智能体 demo。

## 许可证

发布前需要为代码选择许可证。模型权重、数据集和 benchmark assets 遵循各自上游许可证，本仓库不重新分发这些资产。
