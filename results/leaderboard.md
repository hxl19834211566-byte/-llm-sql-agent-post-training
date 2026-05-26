# 结果记录

评估口径：固定 Spider 风格 eval500，指标为 SQLite 执行准确率。

| 方案 | 正确数 | 准确率 | 备注 |
| --- | ---: | ---: | --- |
| Qwen3-4B-Base baseline | 313/500 | 62.6% | 原始模型 |
| SFT v2 chat | 329/500 | 65.8% | 数据量增加后仍有输出边界问题 |
| SFT v3 greedy | 356/500 | 71.2% | 训练后直接贪心解码 |
| DPO v2 | 353/500 | 70.6% | 偏好优化对照 |
| DPO v3 | 353/500 | 70.6% | 偏好优化对照 |
| OPD v1 | 346/500 | 69.2% | teacher distillation 对照 |
| ORPO v1 | 363/500 | 72.6% | preference optimization 对照 |
| SFT v3 + schema v2 prompt-only | 361/500 | 72.2% | 只改 schema prompt |
| SFT v3 + schema v2 + rerank v14 | 407/500 | 81.4% | 未加 value-linking |
| GRPO v1 + schema v2 + value-linking n20 + rerank v14 | 413/500 | 82.6% | RL 候选生成对照 |
| SFT v3 + schema v2 + value-linking n10 + rerank v14 | 415/500 | 83.0% | 加入 value hints |
| SFT v3 + schema v2 + value-linking n20 + rerank v14 | 428/500 | 85.6% | 候选更多，选择器仍有漏选 |
| SFT v3 + schema v2 + value-linking n20 + rerank v15 | 450/500 | 90.0% | 当前采用 |
| SFT v3 + schema v2 + value-linking n20 oracle | 451/500 | 90.2% | 同一批候选能达到的最好结果 |

当前采用结果为 `450/500 = 90.0%`。
