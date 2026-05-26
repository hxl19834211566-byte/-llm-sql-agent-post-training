#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/project}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/hf_cache/models/Qwen3-4B-Base}"
SFT_V3_ADAPTER="${SFT_V3_ADAPTER:-$PROJECT_ROOT/checkpoints/sft/sql_sft_v3_qwen3_4b}"
SQLITE_ROOT="${SQLITE_ROOT:-$PROJECT_ROOT/data/raw/spider/database}"
TABLES_JSON="${TABLES_JSON:-$PROJECT_ROOT/data/raw/spider/tables.json}"

SFT_V3_TRAIN="${SFT_V3_TRAIN:-$PROJECT_ROOT/data/sft/sql_sft_v3_boundary_error_5000.jsonl}"
SCHEMA_V2_INDEX="${SCHEMA_V2_INDEX:-$PROJECT_ROOT/data/processed/spider_schema_v2.json}"
SCHEMA_V2_TRAIN="${SCHEMA_V2_TRAIN:-$PROJECT_ROOT/data/grpo/sql_grpo_v1_schema_v2_train_5000.jsonl}"
GRPO_TRAIN_FILE="${GRPO_TRAIN_FILE:-$PROJECT_ROOT/data/grpo/sql_grpo_v1_schema_v2_value_linking_train_5000.jsonl}"
RUN_TAG="${RUN_TAG:-grpo_v1}"
GRPO_OUTPUT="${GRPO_OUTPUT:-$PROJECT_ROOT/checkpoints/grpo/sql_grpo_v1_${RUN_TAG}_from_sft_v3_qwen3_4b}"

MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-300}"
MAX_STEPS="${MAX_STEPS:-40}"
NUM_GENERATIONS="${NUM_GENERATIONS:-6}"
GRPO_LR="${GRPO_LR:-5e-6}"
SFT_ANCHOR_COEF="${SFT_ANCHOR_COEF:-0.05}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
EVAL_COUNT="${EVAL_COUNT:-500}"
EVAL_LIMIT="${EVAL_LIMIT:-120}"
EVAL_TAG="${EVAL_TAG:-eval${EVAL_COUNT}_limit${EVAL_LIMIT}}"
NUM_CANDIDATES="${NUM_CANDIDATES:-20}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
PROBE_ONLY="${PROBE_ONLY:-0}"
PROBE_SAMPLES="${PROBE_SAMPLES:-2}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"

cd "$PROJECT_ROOT"

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py310

mkdir -p "$PROJECT_ROOT/data/grpo" "$PROJECT_ROOT/logs" "$PROJECT_ROOT/checkpoints/grpo"

if [[ ! -f "$SCHEMA_V2_INDEX" ]]; then
  python scripts/build_spider_schema_v2.py \
    --tables-json "$TABLES_JSON" \
    --database-dir "$SQLITE_ROOT" \
    --output "$SCHEMA_V2_INDEX"
fi

if [[ "$FORCE_PREPARE" == "1" || ! -f "$SCHEMA_V2_TRAIN" ]]; then
  python scripts/upgrade_jsonl_schema_v2.py \
    --input "$SFT_V3_TRAIN" \
    --schema-index "$SCHEMA_V2_INDEX" \
    --output "$SCHEMA_V2_TRAIN" \
    --summary-output "$PROJECT_ROOT/logs/sql_grpo_v1_schema_v2_train_5000_summary.json" \
    --format-version "grpo_v1_schema_v2_train"
fi

if [[ "$FORCE_PREPARE" == "1" || ! -f "$GRPO_TRAIN_FILE" ]]; then
  python scripts/upgrade_jsonl_value_linking.py \
    --input "$SCHEMA_V2_TRAIN" \
    --sqlite-root "$SQLITE_ROOT" \
    --output "$GRPO_TRAIN_FILE" \
    --summary-output "$PROJECT_ROOT/logs/sql_grpo_v1_schema_v2_value_linking_train_5000_summary.json" \
    --format-version "grpo_v1_schema_v2_value_linking_train" \
    --max-mentions 24 \
    --max-hints 8 \
    --min-score 82 \
    --max-rows-per-query 50 \
    --max-row-fields 6
fi

TRAIN_ARGS=(
  scripts/train_sql_grpo.py
  --model-path "$MODEL_PATH" \
  --sft-adapter-path "$SFT_V3_ADAPTER" \
  --train-file "$GRPO_TRAIN_FILE" \
  --sqlite-root "$SQLITE_ROOT" \
  --output-dir "$GRPO_OUTPUT" \
  --max-train-samples "$MAX_TRAIN_SAMPLES" \
  --max-steps "$MAX_STEPS" \
  --num-generations "$NUM_GENERATIONS" \
  --max-new-tokens 128 \
  --max-prompt-tokens 1536 \
  --max-length 1664 \
  --prompt-head-tokens 64 \
  --temperature 0.7 \
  --top-p 0.9 \
  --learning-rate "$GRPO_LR" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --sft-anchor-coef "$SFT_ANCHOR_COEF" \
  --timeout-sec 5 \
  --logging-steps 5 \
  --rollout-log-steps 10 \
  --min-usable-rows 50 \
  --probe-samples "$PROBE_SAMPLES" \
  --seed 42
)

if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
  TRAIN_ARGS+=(--gradient-checkpointing)
fi
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  TRAIN_ARGS+=(--preflight-only)
fi
if [[ "$PROBE_ONLY" == "1" ]]; then
  TRAIN_ARGS+=(--probe-only)
fi

python "${TRAIN_ARGS[@]}" 2>&1 | tee "$PROJECT_ROOT/logs/train_sql_grpo_v1_${RUN_TAG}_qwen3_4b.log"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "PREFLIGHT_ONLY=1, skip model evaluation."
  exit 0
fi

if [[ "$PROBE_ONLY" == "1" ]]; then
  echo "PROBE_ONLY=1, skip training and model evaluation."
  exit 0
fi

EVAL_SCHEMA_V2="$PROJECT_ROOT/data/eval/day2_spider_schema_v2_eval_${EVAL_COUNT}.jsonl"
EVAL_VALUE_LINKING="$PROJECT_ROOT/data/eval/day2_spider_schema_v2_value_linking_eval_${EVAL_COUNT}.jsonl"

if [[ "$FORCE_PREPARE" == "1" || ! -f "$EVAL_SCHEMA_V2" ]]; then
  python scripts/upgrade_jsonl_schema_v2.py \
    --input "$PROJECT_ROOT/data/eval/day2_spider_schema_eval_${EVAL_COUNT}.jsonl" \
    --schema-index "$SCHEMA_V2_INDEX" \
    --output "$EVAL_SCHEMA_V2" \
    --summary-output "$PROJECT_ROOT/logs/day2_spider_schema_v2_eval_${EVAL_COUNT}_summary.json" \
    --format-version "eval_schema_v2_prompt_only"
fi

if [[ "$FORCE_PREPARE" == "1" || ! -f "$EVAL_VALUE_LINKING" ]]; then
  python scripts/upgrade_jsonl_value_linking.py \
    --input "$EVAL_SCHEMA_V2" \
    --sqlite-root "$SQLITE_ROOT" \
    --output "$EVAL_VALUE_LINKING" \
    --summary-output "$PROJECT_ROOT/logs/day2_spider_schema_v2_value_linking_eval_${EVAL_COUNT}_summary.json" \
    --format-version "eval_schema_v2_value_linking_prompt_v1" \
    --max-mentions 24 \
    --max-hints 8 \
    --min-score 82 \
    --max-rows-per-query 50 \
    --max-row-fields 6
fi

GRPO_GREEDY_PRED="$PROJECT_ROOT/logs/grpo_v1_${RUN_TAG}_schema_v2_value_linking_${EVAL_TAG}_greedy_predictions.jsonl"
GRPO_GREEDY_EVAL="$PROJECT_ROOT/logs/grpo_v1_${RUN_TAG}_schema_v2_value_linking_${EVAL_TAG}_greedy_exec_eval.jsonl"
GRPO_GREEDY_SUMMARY="$PROJECT_ROOT/logs/grpo_v1_${RUN_TAG}_schema_v2_value_linking_${EVAL_TAG}_greedy_exec_summary.json"
GRPO_CANDIDATES="$PROJECT_ROOT/logs/grpo_v1_${RUN_TAG}_schema_v2_value_linking_candidates_${EVAL_TAG}_n${NUM_CANDIDATES}.jsonl"
GRPO_SELECTED="$PROJECT_ROOT/logs/grpo_v1_${RUN_TAG}_schema_v2_value_linking_selected_${EVAL_TAG}_n${NUM_CANDIDATES}_v14.jsonl"
GRPO_ANALYSIS="$PROJECT_ROOT/logs/grpo_v1_${RUN_TAG}_schema_v2_value_linking_analysis_${EVAL_TAG}_n${NUM_CANDIDATES}_v14.jsonl"
GRPO_SUMMARY="$PROJECT_ROOT/logs/grpo_v1_${RUN_TAG}_schema_v2_value_linking_summary_${EVAL_TAG}_n${NUM_CANDIDATES}_v14.json"

python scripts/run_sql_completion_predictions.py \
  --model-path "$MODEL_PATH" \
  --adapter-path "$GRPO_OUTPUT" \
  --eval-file "$EVAL_VALUE_LINKING" \
  --output-file "$GRPO_GREEDY_PRED" \
  --max-new-tokens 128 \
  --limit "$EVAL_LIMIT" \
  --stream-write 2>&1 | tee "$PROJECT_ROOT/logs/run_grpo_v1_${RUN_TAG}_schema_v2_value_linking_${EVAL_TAG}_greedy.log"

python eval/sql_exec_eval.py \
  --predictions "$GRPO_GREEDY_PRED" \
  --schema-index "$SCHEMA_V2_INDEX" \
  --output "$GRPO_GREEDY_EVAL" \
  --summary-output "$GRPO_GREEDY_SUMMARY" \
  --timeout-sec 5

python scripts/run_sql_candidate_predictions.py \
  --model-path "$MODEL_PATH" \
  --adapter-path "$GRPO_OUTPUT" \
  --eval-file "$EVAL_VALUE_LINKING" \
  --output-file "$GRPO_CANDIDATES" \
  --max-new-tokens 128 \
  --num-candidates "$NUM_CANDIDATES" \
  --temperature 0.7 \
  --top-p 0.9 \
  --limit "$EVAL_LIMIT" 2>&1 | tee "$PROJECT_ROOT/logs/run_grpo_v1_${RUN_TAG}_schema_v2_value_linking_candidates_${EVAL_TAG}_n${NUM_CANDIDATES}.log"

python scripts/rerank_sql_candidates.py \
  --candidates "$GRPO_CANDIDATES" \
  --schema-index "$SCHEMA_V2_INDEX" \
  --sqlite-root "$SQLITE_ROOT" \
  --selected-output "$GRPO_SELECTED" \
  --analysis-output "$GRPO_ANALYSIS" \
  --summary-output "$GRPO_SUMMARY" \
  --timeout-sec 5

python - "$GRPO_OUTPUT/train_results.json" "$GRPO_GREEDY_SUMMARY" "$GRPO_SUMMARY" <<'PY'
import json
import sys

train = json.load(open(sys.argv[1], encoding="utf-8"))
greedy = json.load(open(sys.argv[2], encoding="utf-8"))
rerank = json.load(open(sys.argv[3], encoding="utf-8"))

print(json.dumps({
    "route": "grpo_v1_from_sft_v3_schema_v2_value_linking",
    "train_steps": train.get("global_steps"),
    "train_samples": train.get("usable_train_rows"),
    "num_generations": train.get("num_generations"),
    "greedy_exec_match": greedy.get("exec_match"),
    "greedy_total": greedy.get("total"),
    "greedy_execution_accuracy": greedy.get("execution_accuracy"),
    "selected_exec_match": rerank.get("selected_exec_match"),
    "oracle_exec_match": rerank.get("oracle_exec_match"),
    "selected_execution_accuracy": rerank.get("selected_execution_accuracy"),
    "oracle_execution_accuracy": rerank.get("oracle_execution_accuracy"),
    "baseline_selected": "428/500",
    "baseline_oracle": "451/500",
    "adopt_if": "oracle_exec_match > 451 and selected_exec_match > 428",
}, ensure_ascii=False, indent=2))
PY
