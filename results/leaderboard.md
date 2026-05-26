# Public Leaderboard

Metric: SQLite execution accuracy on the fixed Spider-style eval500 split.

| System | Correct | Accuracy | Notes |
| --- | ---: | ---: | --- |
| Qwen3-4B-Base baseline | 313/500 | 62.6% | Raw baseline |
| SFT v2 chat | 329/500 | 65.8% | Improved data scale, still output-boundary issues |
| SFT v3 greedy | 356/500 | 71.2% | Best training checkpoint |
| DPO v2 | 353/500 | 70.6% | Preference side route |
| DPO v3 | 353/500 | 70.6% | Preference side route |
| OPD v1 | 346/500 | 69.2% | Teacher-distillation side route |
| ORPO v1 | 363/500 | 72.6% | Preference side route |
| SFT v3 + schema v2 prompt-only | 361/500 | 72.2% | Inference prompt improvement |
| SFT v3 + schema v2 + rerank v14 | 407/500 | 81.4% | No value-linking |
| GRPO v1 + schema v2 + value-linking n20 + rerank v14 | 413/500 | 82.6% | RL candidate-generation side route |
| SFT v3 + schema v2 + value-linking n10 + rerank v14 | 415/500 | 83.0% | Better candidate oracle |
| SFT v3 + schema v2 + value-linking n20 + rerank v14 | 428/500 | 85.6% | Higher oracle, selector still missed cases |
| SFT v3 + schema v2 + value-linking n20 + rerank v15 | 450/500 | 90.0% | Final selected result |
| SFT v3 + schema v2 + value-linking n20 oracle | 451/500 | 90.2% | Candidate upper bound, not deployable accuracy |

The final reported deployable result is `450/500 = 90.0%`.
