# Final Pipeline Runbook

Date: 2026-05-11

Last updated: 2026-05-12

## Goal

复现当前最终 SQL 主线：

```text
SFT v3 + schema v2 candidates + rerank v14
SFT v3 + schema v2 + value-linking v1 n20 candidates + rerank v15
= 450/500
= 90.0% execution accuracy
```

固定原则：

- checkpoint 固定为 `SFT v3`。
- eval 固定为 `eval500`。
- schema prompt 固定为 `schema v2`。
- rerank 固定为当前 `scripts/rerank_sql_candidates.py` 实现。
- 不启用 `--enable-join-graph`。
- 结果以 selected execution accuracy 为准，oracle 只作为上限分析。

## Environment

服务器默认项目目录：

```bash
cd /root/project
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py310
```

默认路径：

```bash
PROJECT_ROOT=/root/project
MODEL_PATH=/root/project/hf_cache/models/Qwen3-4B-Base
SFT_V3_ADAPTER=/root/project/checkpoints/sft/sql_sft_v3_qwen3_4b
```

关键输入：

```text
data/raw/spider/tables.json
data/raw/spider/database
data/eval/day2_spider_schema_eval_500.jsonl
checkpoints/sft/sql_sft_v3_qwen3_4b
```

## Expected Final Output

最终 summary 应为：

```json
{
  "total": 500,
  "greedy_exec_match": 367,
  "greedy_execution_accuracy": 0.734,
  "selected_exec_match": 450,
  "selected_execution_accuracy": 0.9,
  "oracle_exec_match": 451,
  "oracle_execution_accuracy": 0.902,
  "selected_minus_greedy": 83,
  "oracle_minus_greedy": 84,
  "enable_join_graph": false,
  "score_version": "v15"
}
```

允许字段顺序不同，但核心数字必须一致：

```text
selected = 450/500 = 90.0%
oracle = 451/500 = 90.2%
```

## Fast Reproduce From Existing Candidates

如果已经存在候选文件：

```text
logs/rerank_sft_v3_schema_v2_value_linking_candidates_eval500_n20.jsonl
```

可以直接重跑 rerank，不需要 GPU：

```bash
cd /root/project
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py310

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

注意：

```text
不要加 --enable-join-graph。
```

检查结果：

```bash
cat logs/rerank_sft_v3_schema_v2_value_linking_summary_eval500_n20_v15_repro.json
```

预期：

```text
selected_exec_match = 450
oracle_exec_match = 451
enable_join_graph = false
```

## Full Reproduce From SFT v3

如果需要从模型重新生成 candidates，则按下面完整流程执行。

### 1. Build Schema v2

```bash
cd /root/project
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py310

python scripts/build_spider_schema_v2.py \
  --tables-json data/raw/spider/tables.json \
  --database-dir data/raw/spider/database \
  --output data/processed/spider_schema_v2.json
```

### 2. Build Schema v2 Eval500

```bash
python scripts/upgrade_jsonl_schema_v2.py \
  --input data/eval/day2_spider_schema_eval_500.jsonl \
  --schema-index data/processed/spider_schema_v2.json \
  --output data/eval/day2_spider_schema_v2_eval_500.jsonl \
  --summary-output logs/day2_spider_schema_v2_eval_500_summary.json \
  --format-version eval_schema_v2_prompt_only
```

### 3. Generate SFT v3 Schema v2 Candidates

这一步需要 GPU。

先构建 value-linking prompt eval：

```bash
python scripts/upgrade_jsonl_value_linking.py \
  --input data/eval/day2_spider_schema_v2_eval_500.jsonl \
  --sqlite-root data/raw/spider/database \
  --output data/eval/day2_spider_schema_v2_value_linking_eval_500.jsonl \
  --summary-output logs/day2_spider_schema_v2_value_linking_eval_500_summary.json \
  --max-hints 8 \
  --max-mentions 24 \
  --min-score 82 \
  --max-rows-per-query 50
```

```bash
python scripts/run_sql_candidate_predictions.py \
  --model-path hf_cache/models/Qwen3-4B-Base \
  --adapter-path checkpoints/sft/sql_sft_v3_qwen3_4b \
  --eval-file data/eval/day2_spider_schema_v2_value_linking_eval_500.jsonl \
  --output-file logs/rerank_sft_v3_schema_v2_value_linking_candidates_eval500_n20_repro.jsonl \
  --max-new-tokens 128 \
  --num-candidates 20 \
  --temperature 0.7 \
  --top-p 0.9
```

生成策略：

- 第 1 条为 greedy。
- 后续最多 19 条为 sampling。
- 每题最多 20 条去重候选。

由于 sampling 存在随机性，如果脚本和环境没有完全固定随机种子，重新生成的 candidates 可能和历史文件不完全一致。因此最终报告中的 `450/500` 应以已保存的候选文件和 summary 为准；完整复跑用于验证流程，不保证每次采样逐字一致。

### 4. Run Final Rerank / Repair

```bash
python scripts/rerank_sql_candidates.py \
  --candidates logs/rerank_sft_v3_schema_v2_value_linking_candidates_eval500_n20_repro.jsonl \
  --schema-index data/processed/spider_schema_v2.json \
  --sqlite-root data/raw/spider/database \
  --selected-output logs/rerank_sft_v3_schema_v2_value_linking_selected_eval500_n20_v15_repro_full.jsonl \
  --analysis-output logs/rerank_sft_v3_schema_v2_value_linking_analysis_eval500_n20_v15_repro_full.jsonl \
  --summary-output logs/rerank_sft_v3_schema_v2_value_linking_summary_eval500_n20_v15_repro_full.json \
  --timeout-sec 5 \
  --score-version v15
```

再次强调：

```text
不要启用 --enable-join-graph。
```

当前 v15 主线不是 schema graph join reward 路线。

## Reported Historical Artifacts

最终报告采用这些已保存产物：

```text
logs/rerank_sft_v3_schema_v2_candidates_eval500.jsonl
logs/rerank_sft_v3_schema_v2_selected_eval500_v12.jsonl
logs/rerank_sft_v3_schema_v2_analysis_eval500_v12.jsonl
logs/rerank_sft_v3_schema_v2_summary_eval500_v12.json
logs/rerank_sft_v3_schema_v2_summary_eval500_v13.json
logs/rerank_sft_v3_schema_v2_selected_eval500_v14.jsonl
logs/rerank_sft_v3_schema_v2_analysis_eval500_v14.jsonl
logs/rerank_sft_v3_schema_v2_summary_eval500_v14.json
data/eval/day2_spider_schema_v2_value_linking_eval_500.jsonl
logs/rerank_sft_v3_schema_v2_value_linking_candidates_eval500_n10.jsonl
logs/rerank_sft_v3_schema_v2_value_linking_selected_eval500_n10_v14.jsonl
logs/rerank_sft_v3_schema_v2_value_linking_analysis_eval500_n10_v14.jsonl
logs/rerank_sft_v3_schema_v2_value_linking_summary_eval500_n10_v14.json
logs/rerank_sft_v3_schema_v2_value_linking_candidates_eval500_n20.jsonl
logs/rerank_sft_v3_schema_v2_value_linking_selected_eval500_n20_v14.jsonl
logs/rerank_sft_v3_schema_v2_value_linking_analysis_eval500_n20_v14.jsonl
logs/rerank_sft_v3_schema_v2_value_linking_summary_eval500_n20_v14.json
logs/rerank_sft_v3_schema_v2_value_linking_selected_eval500_n20_v15.jsonl
logs/rerank_sft_v3_schema_v2_value_linking_analysis_eval500_n20_v15.jsonl
logs/rerank_sft_v3_schema_v2_value_linking_summary_eval500_n20_v15.json
```

v12 和 v13 的 summary 一致，v14 在此基础上净增 3 题且 0 回退：

```text
v12/v13 selected_exec_match = 404
v14 selected_exec_match = 407
v14 selected_execution_accuracy = 0.814
oracle_exec_match = 409
oracle_execution_accuracy = 0.818
```

value-linking v1 n10 在 v14 基础上进一步提升：

```text
selected_exec_match = 415
selected_execution_accuracy = 0.83
oracle_exec_match = 436
oracle_execution_accuracy = 0.872
```

value-linking v1 n20 继续提升：

```text
selected_exec_match = 428
selected_execution_accuracy = 0.856
oracle_exec_match = 451
oracle_execution_accuracy = 0.902
```

rerank / repair v15 在同一批 value-linking n20 candidates 上进一步提升：

```text
selected_exec_match = 450
selected_execution_accuracy = 0.9
oracle_exec_match = 451
oracle_execution_accuracy = 0.902
```

## Comparison Checks

旧 schema v1 最佳：

```text
logs/rerank_sft_v3_summary_eval500_v8.json
```

结果：

```text
SFT v3 + schema v1 + rerank v8
= 384/500
= 76.8%
```

schema v2 final：

```text
SFT v3 + schema v2 + value-linking v1 n20 + rerank v15
= 450/500
= 90.0%
```

净提升：

```text
450 - 384 = +66
76.8% -> 90.0%
```

注意边界：

```text
value-linking v1 n20 + rerank v15 是 schema v2 prompt candidates 专用。
schema v1 candidates 不应改用 v15 替代 v8。
```

## Troubleshooting

### Result Below 450

优先检查：

1. 是否使用了历史 candidates 文件。
2. 是否 schema index 指向 `data/processed/spider_schema_v2.json`。
3. 是否 eval 文件是 `day2_spider_schema_v2_eval_500.jsonl`。
4. 是否没有启用 `--enable-join-graph`。
5. 是否 SQLite root 指向 `data/raw/spider/database`。
6. 是否脚本版本为最新 `scripts/rerank_sql_candidates.py`。

### Full Reproduce Not Exactly 450

如果是重新生成 candidates 后结果不等于 428，先不要直接否定主结果。原因是 candidate generation 使用 sampling：

```text
temperature = 0.7
top_p = 0.9
num_candidates = 20
```

历史主结果固定在已保存 candidates 文件上。若要完全可重复，需要在候选生成脚本中新增随机种子控制，然后重新生成一版命名为 locked candidates 的产物。

### Oracle Is Not Final Accuracy

报告中只能把下面数字作为最终准确率：

```text
selected_exec_match = 450/500
```

下面数字只能作为候选集理论上限：

```text
oracle_exec_match = 451/500
```

## Next Step After Freezing

当前主线已经固定。后续优先事项：

1. 在最终项目报告中引用 `docs/final_mainline_report.md`。
2. 如果继续扩展，应优先提升候选生成质量或加入 execution-guided repair。
3. 如果继续提升 SQL 分数，先做更强候选生成或 execution-guided repair，不再继续堆固定 eval500 硬规则。
