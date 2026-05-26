# Value Linking v1 Report

Date: 2026-05-12

## Goal

当前主线已经接近原候选集 oracle：

```text
SFT v3 + schema v2 candidates + rerank v14
selected = 407/500 = 81.4%
oracle   = 409/500 = 81.8%
```

这说明继续只调 rerank 的空间很小，下一步应提高 candidate oracle upper bound。

Value Linking v1 的目标是验证：

```text
在生成 SQL 前，把 SQLite 中检索到的真实 value / id / abbreviation hints 加入 prompt，
能否让候选集中出现更多正确 SQL。
```

## Method

这是 prompt-only / inference-time 方法，不训练模型：

```text
question
-> rule-based mention extraction
-> SQLite text-column retrieval under current db_id
-> same-row ID / abbreviation / name hints
-> append hints to schema v2 prompt
-> SFT v3 generates candidates
-> rerank v14 selects SQL
```

脚本：

```text
scripts/upgrade_jsonl_value_linking.py
```

生成的 eval 文件：

```text
data/eval/day2_spider_schema_v2_value_linking_eval_500.jsonl
```

Value hints summary：

```text
rows = 500
rows_with_hints = 181
rows_without_hints = 319
hints_total = 551
```

Top db hints：

| db | hints |
| --- | ---: |
| `wta_1` | 181 |
| `flight_2` | 158 |
| `cre_Doc_Template_Mgt` | 80 |
| `car_1` | 71 |
| `concert_singer` | 31 |

## Quality Check

关键样本命中：

```text
France -> countries.CountryName = "france"; CountryId = 3; Continent = 2
European -> continents.Continent = "europe"; ContId = 2
Jetblue Airways -> airlines.Airline = "JetBlue Airways"; uid = 8; Abbreviation = "JetBlue"
```

生成前曾发现 `Airways` 作为单独 mention 会带入 `US Airways` / `AirTran Airways` 噪声，已在脚本中过滤 generic value token。

## Result

Candidate generation：

```text
model = Qwen3-4B-Base
adapter = checkpoints/sft/sql_sft_v3_qwen3_4b
eval = data/eval/day2_spider_schema_v2_value_linking_eval_500.jsonl
num_candidates = 10
temperature = 0.7
top_p = 0.9
```

Rerank：

```text
rerank = v14
enable_join_graph = false
```

Summary：

| Pipeline | Greedy | Selected | Oracle |
| --- | ---: | ---: | ---: |
| SFT v3 + schema v2 + rerank v14 | 361/500 | 407/500 | 409/500 |
| SFT v3 + schema v2 + value linking v1 n10 + rerank v14 | 367/500 | 415/500 | 436/500 |
| SFT v3 + schema v2 + value linking v1 n20 + rerank v14 | 367/500 | 428/500 | 451/500 |

Delta n10 vs no-value-linking：

```text
selected: +8
oracle:   +27
greedy:   +6
```

Delta n20 vs n10：

```text
selected: +13
oracle:   +15
greedy:   +0
```

This satisfies the planned success condition:

```text
selected_exec_match > 407
oracle_exec_match > 409
```

## Transfer Analysis

Compared with the previous v14 mainline:

```text
selected improved: 29
selected regressed: 21
net selected: +8
oracle improved: 37
oracle regressed: 10
net oracle: +27
```

Representative improvements:

| id | db | fix |
| --- | --- | --- |
| `day2_spider_schema_val_0114` | `car_1` | `france` linked to `countries.CountryName`, enabling correct country join |
| `day2_spider_schema_val_0131` | `car_1` | `European` linked to `continents.Continent = europe`, `ContId = 2` |
| `day2_spider_schema_val_0183` | `flight_2` | `Jetblue Airways` linked to exact database value `JetBlue Airways` |
| `day2_spider_schema_val_0215` | `flight_2` | same JetBlue value link fixes flight count |
| `day2_spider_schema_val_0304` | `cre_Doc_Template_Mgt` | document name value helps select the right document/template join |
| `day2_spider_schema_val_0357` | `cre_Doc_Template_Mgt` | `Presentation` value link helps template type lookup |

Representative regressions:

| pattern | example |
| --- | --- |
| prompt hints changed sampling distribution and produced a worse join | hometown teacher count joined `course_arrange` unnecessarily |
| candidate pool changed column order / projection | paragraph count by document id selected count first |
| value hints nudged model toward over-specific or wrong value joins | some WTA tournament/year questions |
| more candidates exposed plausible but wrong SQL that rerank v14 selected | airport fewest flights / no in-out flights |

## Decision

Value Linking v1 is a successful accuracy-improvement route. The current best is n20:

```text
selected = 428/500 = 85.6%
oracle   = 451/500 = 90.2%
```

n10 regression profile:

```text
+29 selected fixes
-21 selected regressions
net +8
```

n20 compared with n10:

```text
selected improved: 24
selected regressed: 11
net selected: +13
oracle improved: 23
oracle regressed: 8
net oracle: +15
```

The larger oracle gain is the most important signal:

```text
409 -> 451
```

This proves the previous bottleneck was candidate generation, not rerank.

## Next Step

Do not train yet.

Recommended next experiments:

1. Improve selection for n20 with a value-linking-aware rerank v15.
2. Analyze the 11 n20 regressions versus n10 and the remaining 23 oracle gap.
3. Add execution-guided repair only for remaining oracle-miss rows.

Adoption rule remains:

```text
Only replace the final mainline if selected_exec_match beats the current best.
Current best after this experiment: 428/500.
```
