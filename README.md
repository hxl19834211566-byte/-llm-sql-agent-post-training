# Text-to-SQL Agent：SFT + Value Linking + Rerank

这是一个面向 BI / Text-to-SQL 场景的 LLM SQL Agent 项目，核心评估指标是 SQLite 执行准确率。最终系统使用 Qwen3-4B-Base 的 LoRA SFT checkpoint，结合 schema-aware prompt、value-linking hints、多候选生成，以及确定性的 rerank / repair 阶段。

最终主线结果：

```text
SFT v3 + schema v2 + value-linking v1 + n20 candidates + rerank v15
= 450 / 500
= 90.0% SQLite execution accuracy
```

对应候选集 oracle upper bound 为：

```text
451 / 500 = 90.2%
```

`90.0%` 是最终可报告的 selected accuracy；`90.2%` 只表示当前 n20 候选集的理论上限。

## 内容

- 核心 SFT、推理和执行评估脚本。
- DPO / OPD / ORPO / GRPO 可选副线实验脚本。
- Spider 风格 SQLite 数据库上的 SQL execution evaluation。
- schema v2 prompt 构建与 value-linking prompt augmentation。
- 多候选 SQL 生成。
- SQL candidate rerank / repair。
- 工具调用式 SQL Agent CLI demo。
- 最终报告和结果摘要。

以下本地资产不会提交到 Git：

- Hugging Face base model cache。
- LoRA adapter 权重。
- Spider SQLite database。
- 生成的 JSONL 数据、候选 SQL、prediction dump 和原始训练/评估数据。
- 本地笔记和临时产物。

## 目录结构

```text
demo/        工具调用式 SQL Agent demo
docs/        主线报告、副线说明和复现 runbook
eval/        SQLite execution evaluator
results/     结果摘要
scripts/     数据处理、训练、生成、value-linking、rerank 脚本
data/        本地数据工作区，大部分内容被 gitignore
checkpoints/ 本地 checkpoint 工作区，权重被 gitignore
```

## 最终 Pipeline

```text
Qwen3-4B-Base
-> SFT v3 LoRA
-> schema v2 prompt
-> value-linking v1 prompt hints
-> 每题生成 20 个 SQL candidates
-> rerank / repair v15
-> SQLite execution evaluation
```

SFT v3 主要解决输出边界问题，让模型稳定输出一条只读 SQL。后续主要错误集中在 schema linking、value grounding、JOIN、别名、聚合和返回列语义上，因此最终提升主要来自候选生成和 rerank，而不是继续盲目训练。

## 核心结果

| System | Execution accuracy |
| --- | ---: |
| Qwen3-4B-Base baseline | 313/500 = 62.6% |
| SFT v3 greedy | 356/500 = 71.2% |
| SFT v3 + schema v2 prompt-only | 361/500 = 72.2% |
| SFT v3 + schema v2 + rerank v14 | 407/500 = 81.4% |
| SFT v3 + schema v2 + value-linking n20 + rerank v14 | 428/500 = 85.6% |
| SFT v3 + schema v2 + value-linking n20 + rerank v15 | 450/500 = 90.0% |

完整结果见：

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

## 复现最终 Rerank

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

预期核心结果：

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

最终报告结果不要添加 `--enable-join-graph`。

## 运行 Demo

列出本地 candidate 文件中的样例：

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
  --row-id day2_spider_schema_val_0001 \
  --candidates-file logs/rerank_sft_v3_schema_v2_value_linking_candidates_eval500_n20.jsonl \
  --schema-index data/processed/spider_schema_v2.json \
  --sqlite-root data/raw/spider/database \
  --trace-output logs/sql_agent_demo_trace.json
```

demo 会输出：

```text
search_schema
model_sql
validate_sql
run_sql
final_answer
```

## 文档入口

- [docs/mainline_report.md](docs/mainline_report.md)：最终主线结果和采用理由。
- [docs/final_pipeline_runbook.md](docs/final_pipeline_runbook.md)：完整复现流程。
- [docs/side_routes.md](docs/side_routes.md)：DPO / OPD / ORPO / GRPO 副线实验。
- [docs/value_linking_v1_report.md](docs/value_linking_v1_report.md)：value-linking v1 的方法和收益。
- [docs/rerank_repair_report.md](docs/rerank_repair_report.md)：rerank / repair v15 的改进。
- [docs/tool_calling_demo_report.md](docs/tool_calling_demo_report.md)：工具调用式 SQL Agent demo。

## License

发布前需要为代码选择许可证。模型权重、数据集和 benchmark assets 遵循各自上游许可证，本仓库不重新分发这些资产。
