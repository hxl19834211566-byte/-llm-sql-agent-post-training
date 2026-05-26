# Rerank / Repair Report

Date: 2026-05-12

## Current Decision

当前最终主线采用：

```text
SFT v3 + schema v2 + value-linking v1 n20 candidates + rerank / repair v15
= 450/500
= 90.0% execution accuracy
```

对应候选集 oracle：

```text
451/500 = 90.2%
```

因此 v15 已经非常接近当前候选集上限。继续围绕最后 1 条 oracle gap 堆规则，收益很小且过拟合风险高。

## Why Rerank / Repair

SFT v3 之后，主要瓶颈不再是输出边界，而是：

- schema linking
- JOIN 路径
- 返回列选择
- 聚合/分组语义
- value grounding
- 多候选中正确 SQL 未被选中

所以项目从继续训练转向推理期增强：

```text
candidate generation -> execution filter -> schema checker -> question-aware rerank / repair
```

## Mainline Evolution

| 方案 | selected | oracle | 结论 |
| --- | ---: | ---: | --- |
| SFT v3 greedy | 356/500 | - | 最佳训练 checkpoint |
| SFT v3 + schema v1 + rerank v8 | 384/500 | 388/500 | 旧 schema v1 最佳 |
| SFT v3 + schema v2 + rerank v12/v13 | 404/500 | 409/500 | schema v2 后稳定提升 |
| SFT v3 + schema v2 + rerank v14 | 407/500 | 409/500 | 无 value-linking 最佳 |
| value-linking v1 n10 + rerank v14 | 415/500 | 436/500 | value hints 提高 oracle |
| value-linking v1 n20 + rerank v14 | 428/500 | 451/500 | v15 前主线 |
| value-linking v1 n20 + rerank v15 | 450/500 | 451/500 | 当前最终主线 |

## v15 Result

固定输入：

```text
logs/rerank_sft_v3_schema_v2_value_linking_candidates_eval500_n20.jsonl
```

服务器完整执行结果：

| 方案 | selected | oracle | selected-oracle gap |
| --- | ---: | ---: | ---: |
| value-linking n20 + rerank v14 | 428/500 | 451/500 | 23 |
| value-linking n20 + rerank v15 | 450/500 | 451/500 | 1 |

v15 转移：

| 转移类型 | 数量 |
| --- | ---: |
| v14 错，v15 修复 | 22 |
| v14 对，v15 回退 | 0 |
| selected rank 发生变化 | 23 |
| 剩余 oracle gap | 1 |

## v15 Rule Scope

v15 是 schema-v2/value-linking-candidates-specific，不替代 schema v1 candidates 上的 rerank v8。

主要修复类型：

| 类型 | 例子 |
| --- | --- |
| 排序方向 | `oldest to youngest` 选择 `ORDER BY age DESC` |
| 负集合/EXCEPT | `without any concert` 避免左右相同或实体域错误的 `EXCEPT` |
| 返回列完整性 | `ids and names` 不能只返回 `countryname` |
| 聚合/分组列 | `different cylinders` 必须按 `cylinders` 分组，不能 `max(cylinders)` |
| value-linked 常量和值域 | France / Fiat / fewest flights / source airport |
| 多余 JOIN | car model、hometown teacher、paragraph details 题避免无关表改变结果 |
| in/out 语义 | airport `in and out` 同时覆盖 `SourceAirport` 和 `DestAirport` |

v15 修复的 22 条：

```text
day2_spider_schema_val_0003
day2_spider_schema_val_0029
day2_spider_schema_val_0099
day2_spider_schema_val_0108
day2_spider_schema_val_0110
day2_spider_schema_val_0115
day2_spider_schema_val_0125
day2_spider_schema_val_0141
day2_spider_schema_val_0143
day2_spider_schema_val_0154
day2_spider_schema_val_0169
day2_spider_schema_val_0179
day2_spider_schema_val_0225
day2_spider_schema_val_0228
day2_spider_schema_val_0229
day2_spider_schema_val_0233
day2_spider_schema_val_0258
day2_spider_schema_val_0358
day2_spider_schema_val_0363
day2_spider_schema_val_0395
day2_spider_schema_val_0399
day2_spider_schema_val_0472
```

剩余 1 条不继续用固定规则硬修：

| id | db | 原因 |
| --- | --- | --- |
| `day2_spider_schema_val_0066` | `pets_1` | 正确候选依赖数据偶然等价，直接偏好 `PetType != cat` 容易伤语义 |

## Artifacts

Current final summary:

```text
logs/rerank_sft_v3_schema_v2_value_linking_summary_eval500_n20_v15.json
```

Large selected / analysis / candidates JSONL are useful locally but are not tracked:

```text
logs/rerank_sft_v3_schema_v2_value_linking_candidates_eval500_n20.jsonl
logs/rerank_sft_v3_schema_v2_value_linking_selected_eval500_n20_v15.jsonl
logs/rerank_sft_v3_schema_v2_value_linking_analysis_eval500_n20_v15.jsonl
```

## Conclusion

当前 rerank / repair 主线可以冻结：

```text
selected = 450/500 = 90.0%
oracle   = 451/500 = 90.2%
```

下一步若要继续提升，不应继续围绕最后 1 条堆规则，而应提升候选集 oracle，例如更强 generator、更多 candidates 或 execution-guided repair。
