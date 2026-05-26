#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/project}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/hf_cache/models/Qwen3-4B-Base}"
SFT_V3_ADAPTER="${SFT_V3_ADAPTER:-$PROJECT_ROOT/checkpoints/sft/sql_sft_v3_qwen3_4b}"
ORPO_TRAIN_FILE="${ORPO_TRAIN_FILE:-$PROJECT_ROOT/data/dpo/sql_dpo_v3_train_checked.jsonl}"
ORPO_EVAL_FILE="${ORPO_EVAL_FILE:-$PROJECT_ROOT/data/dpo/sql_dpo_v3_heldout.jsonl}"
ORPO_OUTPUT="${ORPO_OUTPUT:-$PROJECT_ROOT/checkpoints/orpo/sql_orpo_v1_from_sft_v3_qwen3_4b}"
ORPO_ALPHA="${ORPO_ALPHA:-0.05}"
ORPO_LR="${ORPO_LR:-2e-6}"
EVAL_COUNT="${EVAL_COUNT:-500}"
RUN_ORPO_RERANK="${RUN_ORPO_RERANK:-0}"

cd "$PROJECT_ROOT"

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py310

SCHEMA_V2_INDEX="$PROJECT_ROOT/data/processed/spider_schema_v2.json"
EVAL_V2="$PROJECT_ROOT/data/eval/spider_schema_v2_eval_${EVAL_COUNT}.jsonl"
ORPO_EVAL_PRED="$PROJECT_ROOT/logs/orpo_v1_schema_v2_eval${EVAL_COUNT}_predictions.jsonl"
ORPO_EVAL_OUTPUT="$PROJECT_ROOT/logs/orpo_v1_schema_v2_eval${EVAL_COUNT}_exec_eval.jsonl"
ORPO_EVAL_SUMMARY="$PROJECT_ROOT/logs/orpo_v1_schema_v2_eval${EVAL_COUNT}_exec_summary.json"

if [[ ! -f "$SCHEMA_V2_INDEX" ]]; then
  python scripts/build_spider_schema_v2.py \
    --tables-json "$PROJECT_ROOT/data/raw/spider/tables.json" \
    --database-dir "$PROJECT_ROOT/data/raw/spider/database" \
    --output "$SCHEMA_V2_INDEX"
fi

if [[ ! -f "$EVAL_V2" ]]; then
  python scripts/upgrade_jsonl_schema_v2.py \
    --input "$PROJECT_ROOT/data/eval/spider_schema_eval_${EVAL_COUNT}.jsonl" \
    --schema-index "$SCHEMA_V2_INDEX" \
    --output "$EVAL_V2" \
    --summary-output "$PROJECT_ROOT/logs/spider_schema_v2_eval_${EVAL_COUNT}_summary.json" \
    --format-version "eval_schema_v2_prompt_only"
fi

python scripts/train_sql_orpo.py \
  --model-path "$MODEL_PATH" \
  --sft-adapter-path "$SFT_V3_ADAPTER" \
  --train-file "$ORPO_TRAIN_FILE" \
  --eval-file "$ORPO_EVAL_FILE" \
  --output-dir "$ORPO_OUTPUT" \
  --max-length 1536 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate "$ORPO_LR" \
  --warmup-steps 5 \
  --weight-decay 0.0 \
  --orpo-alpha "$ORPO_ALPHA" \
  --logging-steps 5 \
  --eval-steps 50 \
  --seed 42 2>&1 | tee "$PROJECT_ROOT/logs/train_sql_orpo_v1_qwen3_4b.log"

python scripts/run_sql_completion_predictions.py \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ORPO_OUTPUT" \
  --eval-file "$EVAL_V2" \
  --output-file "$ORPO_EVAL_PRED" \
  --max-new-tokens 128 \
  --stream-write 2>&1 | tee "$PROJECT_ROOT/logs/run_orpo_v1_schema_v2_eval${EVAL_COUNT}_predictions.log"

python eval/sql_exec_eval.py \
  --predictions "$ORPO_EVAL_PRED" \
  --schema-index "$SCHEMA_V2_INDEX" \
  --output "$ORPO_EVAL_OUTPUT" \
  --summary-output "$ORPO_EVAL_SUMMARY" \
  --timeout-sec 5

if [[ "$RUN_ORPO_RERANK" == "1" ]]; then
  ORPO_CANDIDATES="$PROJECT_ROOT/logs/orpo_v1_schema_v2_candidates_eval${EVAL_COUNT}.jsonl"
  ORPO_RERANK_SELECTED="$PROJECT_ROOT/logs/orpo_v1_schema_v2_selected_eval${EVAL_COUNT}.jsonl"
  ORPO_RERANK_ANALYSIS="$PROJECT_ROOT/logs/orpo_v1_schema_v2_analysis_eval${EVAL_COUNT}.jsonl"
  ORPO_RERANK_SUMMARY="$PROJECT_ROOT/logs/orpo_v1_schema_v2_summary_eval${EVAL_COUNT}.json"

  python scripts/run_sql_candidate_predictions.py \
    --model-path "$MODEL_PATH" \
    --adapter-path "$ORPO_OUTPUT" \
    --eval-file "$EVAL_V2" \
    --output-file "$ORPO_CANDIDATES" \
    --max-new-tokens 128 \
    --num-candidates 5 \
    --temperature 0.7 \
    --top-p 0.9

  python scripts/rerank_sql_candidates.py \
    --candidates "$ORPO_CANDIDATES" \
    --schema-index "$SCHEMA_V2_INDEX" \
    --sqlite-root "$PROJECT_ROOT/data/raw/spider/database" \
    --selected-output "$ORPO_RERANK_SELECTED" \
    --analysis-output "$ORPO_RERANK_ANALYSIS" \
    --summary-output "$ORPO_RERANK_SUMMARY" \
    --timeout-sec 5
fi

python - "$ORPO_OUTPUT/train_results.json" "$ORPO_EVAL_SUMMARY" <<'PY'
import json
import sys

train = json.load(open(sys.argv[1], encoding="utf-8"))
eval_summary = json.load(open(sys.argv[2], encoding="utf-8"))
print(json.dumps({
    "route": "orpo_v1_from_sft_v3_dpo_v3_pairs",
    "train_rows": train.get("train_rows"),
    "eval_rows": train.get("eval_rows"),
    "orpo_alpha": train.get("orpo_alpha"),
    "learning_rate": train.get("learning_rate"),
    "orpo_eval_exec_match": eval_summary.get("exec_match"),
    "orpo_eval_total": eval_summary.get("total"),
    "orpo_eval_execution_accuracy": eval_summary.get("execution_accuracy"),
    "baseline_sft_v3_schema_v2_prompt_only": "361/500",
    "baseline_dpo_v3": "353/500",
    "best_rerank_baseline": "404/500",
}, ensure_ascii=False, indent=2))
PY
