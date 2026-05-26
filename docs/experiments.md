# 实验记录

这个项目最后采用的是推理期增强路线，而不是继续堆训练副线。下面只保留对理解主线有帮助的结果。

## 主线结果

| 方案 | 正确数 | 执行准确率 | 说明 |
| --- | ---: | ---: | --- |
| Qwen3-4B-Base baseline | 313/500 | 62.6% | 原始基线 |
| SFT v2 chat | 329/500 | 65.8% | 扩大 SFT 后仍有输出边界问题 |
| SFT v3 greedy | 356/500 | 71.2% | 最佳训练 checkpoint |
| SFT v3 + schema v2 prompt-only | 361/500 | 72.2% | schema prompt 小幅提升 |
| SFT v3 + schema v2 + rerank v14 | 407/500 | 81.4% | 无 value-linking 的推理期结果 |
| SFT v3 + schema v2 + value-linking n10 + rerank v14 | 415/500 | 83.0% | value hints 提升候选上限 |
| SFT v3 + schema v2 + value-linking n20 + rerank v14 | 428/500 | 85.6% | 候选更多，但选择器仍有缺口 |
| SFT v3 + schema v2 + value-linking n20 + rerank v15 | 450/500 | 90.0% | 最终主线 |

`451/500 = 90.2%` 是当前 n20 候选集 oracle 上限，不作为最终准确率报告。

## 为什么不继续盲目训练

SFT v3 之后，模型已经能稳定输出只读 SQL。剩下的问题主要是：

- 表和列链接错误；
- value grounding 错误；
- JOIN 路径错误；
- 聚合、排序、返回列语义错误；
- 候选中有正确 SQL，但初始候选没有排到第一。

这些问题更适合用 `schema v2 + value-linking + 多候选 + rerank/repair` 处理。

## 副线实验

这些副线都保留下来作为对照，但没有替代最终主线。

| 副线 | 目的 | 结果 | 结论 |
| --- | --- | ---: | --- |
| DPO v2 | 用更窄的高置信 preference pairs 做偏好优化 | 353/500 = 70.6% | 不采用 |
| DPO v3 | 混合高置信错误类型的偏好优化 | 353/500 = 70.6% | 不采用 |
| OPD v1 | 用 Qwen3-8B teacher 的执行正确 SQL 蒸馏 SFT v3 | 346/500 = 69.2% | 不采用 |
| ORPO v1 | 更轻量的 preference optimization | 363/500 = 72.6% | 不采用 |
| GRPO v1 | 用 execution reward 优化候选生成 | selected 413/500 = 82.6% | 不采用 |

副线对应脚本：

```text
DPO:
  scripts/prepare_sql_dpo_v1.py
  scripts/prepare_sql_dpo_v2_from_v1.py
  scripts/prepare_sql_dpo_v3_from_v1.py
  scripts/validate_sql_dpo_v1.py
  scripts/train_sql_dpo.py

OPD:
  scripts/prepare_opd_teacher_prompts.py
  scripts/build_opd_distill_data.py
  scripts/train_sql_opd_completion.py
  scripts/run_opd_v1.sh

ORPO:
  scripts/train_sql_orpo.py
  scripts/run_orpo_v1.sh

GRPO:
  scripts/train_sql_grpo.py
  scripts/run_grpo_v1.sh
```
