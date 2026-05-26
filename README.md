# Text-to-SQL 智能体：SFT、值链接与重排序

这是一个面向 BI / Text-to-SQL 场景的 SQL Agent 项目。输入自然语言问题、数据库 schema 和 `db_id`，输出一条只读 SQLite SQL，并用执行结果评估是否正确。

当前采用的流程是：

```text
Qwen3-4B-Base
-> SFT v3 LoRA
-> schema v2 prompt
-> value-linking v1
-> 每题生成 20 个 SQL candidates
-> rerank / repair v15
-> SQLite execution evaluation
```

在固定 eval500 上，当前结果为：

```text
450/500 = 90.0% SQLite 执行准确率
```

同一批 n20 候选中最多可命中 `451/500 = 90.2%`，说明剩余主要问题已经集中在候选选择而不是候选生成。

## 做了什么

- 用 Qwen3-4B-Base 做 SQL completion 风格 SFT，稳定输出只读 SQL。
- 构建 schema v2 prompt，补充主键、外键、列名和表结构信息。
- 从 SQLite 数据库抽取 value hints，缓解实体值对齐问题。
- 生成多条候选 SQL，再用规则化 rerank / repair 选择输出 SQL。
- 保留 DPO / OPD / ORPO / GRPO 训练副线，作为和推理期增强的对照。
- 提供离线 SQL Agent demo，展示 `search_schema -> model_sql -> validate_sql -> run_sql -> final_answer` 流程。

## 结果

| 方案 | 执行准确率 |
| --- | ---: |
| Qwen3-4B-Base baseline | 313/500 = 62.6% |
| SFT v3 greedy | 356/500 = 71.2% |
| SFT v3 + schema v2 prompt-only | 361/500 = 72.2% |
| SFT v3 + schema v2 + rerank v14 | 407/500 = 81.4% |
| SFT v3 + schema v2 + value-linking n20 + rerank v14 | 428/500 = 85.6% |
| SFT v3 + schema v2 + value-linking n20 + rerank v15 | 450/500 = 90.0% |

完整实验对照见 [results/leaderboard.md](results/leaderboard.md) 和 [docs/experiments.md](docs/experiments.md)。

## 目录

```text
demo/        SQL Agent demo
docs/        复现、实验记录和 demo 用法
eval/        SQLite 执行评估器
results/     结果摘要
scripts/     数据处理、训练、生成、值链接和重排序脚本
```

本地运行时还会用到这些目录，但仓库不提交其中的大文件：

```text
data/        Spider 数据、schema index、训练 / 评估 JSONL
checkpoints/ LoRA adapter
hf_cache/    base model cache
logs/        生成结果、候选 SQL 和评估日志
```

## 环境

实验环境：

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

需要本地准备：

```text
hf_cache/models/Qwen3-4B-Base
checkpoints/sft/sql_sft_v3_qwen3_4b
data/raw/spider/tables.json
data/raw/spider/database/<db_id>/<db_id>.sqlite
```

这些模型权重、数据库和生成文件体积较大，且受上游许可证约束，因此不随仓库发布。

## 文档

- [docs/reproduce.md](docs/reproduce.md)：复现 90.0% 结果。
- [docs/experiments.md](docs/experiments.md)：主要实验和 DPO / OPD / ORPO / GRPO 副线。
- [docs/demo.md](docs/demo.md)：SQL Agent demo 用法。

## 许可说明

本仓库不重新分发模型权重、数据集和 benchmark assets；这些资源需要按各自上游许可证自行获取。
