# 文档索引

这里存放 Text-to-SQL 智能体项目的报告和复现说明。

## 核心报告

- [主线报告](mainline_report.md)：最终结果、pipeline 和采用理由。
- [最终 pipeline 复现手册](final_pipeline_runbook.md)：重建 schema v2、生成值链接提示、生成候选 SQL、运行 rerank v15 的命令。
- [副线实验](side_routes.md)：DPO、OPD、ORPO、GRPO 的实验目的、脚本、结果和采用判断。

## 主要改进

- [值链接 v1 报告](value_linking_v1_report.md)：SQLite value hints 如何提高候选集 oracle 上限。
- [值链接 v1 复现手册](value_linking_v1_runbook.md)：生成 value-linking eval 文件和 candidates 的命令。
- [重排序 / 修复报告](rerank_repair_report.md)：rerank / repair 如何缩小 selected 和 oracle 的差距。
- [schema v2 逻辑检查](schema_v2_logic_review.md)：schema prompt 构造和校验记录。

## Agent Demo

- [工具调用 demo 报告](tool_calling_demo_report.md)：离线和在线 demo 流程，包括 `search_schema`、`validate_sql`、`run_sql`。

生成的 PDF 和原始日志不进入 Git 跟踪范围。
