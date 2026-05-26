#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/project}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/hf_cache/models/Qwen3-4B-Base}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-$PROJECT_ROOT/hf_cache/models/Qwen3-8B}"
SFT_V3_ADAPTER="${SFT_V3_ADAPTER:-$PROJECT_ROOT/checkpoints/sft/sql_sft_v3_qwen3_4b}"
OPD_SOURCE_LIMIT="${OPD_SOURCE_LIMIT:-2000}"
OPD_HELDOUT_SIZE="${OPD_HELDOUT_SIZE:-120}"
EVAL_COUNT="${EVAL_COUNT:-500}"
RUN_OPD_RERANK="${RUN_OPD_RERANK:-0}"

cd "$PROJECT_ROOT"

source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate py310

SCHEMA_V2_INDEX="$PROJECT_ROOT/data/processed/spider_schema_v2.json"
SFT_V4_SCHEMA_V2="$PROJECT_ROOT/data/sft/sql_sft_v4_schema_v2_5000.jsonl"
EVAL_V2="$PROJECT_ROOT/data/eval/spider_schema_v2_eval_${EVAL_COUNT}.jsonl"

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

if [[ ! -f "$SFT_V4_SCHEMA_V2" ]]; then
  python scripts/upgrade_jsonl_schema_v2.py \
    --input "$PROJECT_ROOT/data/sft/sql_sft_v3_boundary_error_5000.jsonl" \
    --schema-index "$SCHEMA_V2_INDEX" \
    --output "$SFT_V4_SCHEMA_V2" \
    --summary-output "$PROJECT_ROOT/logs/sql_sft_v4_schema_v2_5000_summary.json" \
    --format-version "sft_v4_schema_v2_pk_fk"
fi

OPD_PROMPTS="$PROJECT_ROOT/data/opd/sql_opd_v1_teacher_prompts_${OPD_SOURCE_LIMIT}.jsonl"
OPD_PROMPTS_SUMMARY="$PROJECT_ROOT/logs/sql_opd_v1_teacher_prompts_${OPD_SOURCE_LIMIT}_summary.json"
TEACHER_PRED="$PROJECT_ROOT/logs/opd_v1_qwen3_8b_teacher_train_${OPD_SOURCE_LIMIT}_predictions.jsonl"
TEACHER_EVAL="$PROJECT_ROOT/logs/opd_v1_qwen3_8b_teacher_train_${OPD_SOURCE_LIMIT}_exec_eval.jsonl"
TEACHER_EVAL_SUMMARY="$PROJECT_ROOT/logs/opd_v1_qwen3_8b_teacher_train_${OPD_SOURCE_LIMIT}_exec_summary.json"
OPD_TRAIN="$PROJECT_ROOT/data/opd/sql_opd_v1_train.jsonl"
OPD_HELDOUT="$PROJECT_ROOT/data/opd/sql_opd_v1_heldout.jsonl"
OPD_DATA_SUMMARY="$PROJECT_ROOT/logs/sql_opd_v1_data_summary.json"
OPD_OUTPUT="$PROJECT_ROOT/checkpoints/opd/sql_opd_v1_from_sft_v3_qwen3_4b"
OPD_EVAL_PRED="$PROJECT_ROOT/logs/opd_v1_schema_v2_eval${EVAL_COUNT}_predictions.jsonl"
OPD_EVAL_OUTPUT="$PROJECT_ROOT/logs/opd_v1_schema_v2_eval${EVAL_COUNT}_exec_eval.jsonl"
OPD_EVAL_SUMMARY="$PROJECT_ROOT/logs/opd_v1_schema_v2_eval${EVAL_COUNT}_exec_summary.json"

python scripts/prepare_opd_teacher_prompts.py \
  --input "$SFT_V4_SCHEMA_V2" \
  --exclude-eval-file "$EVAL_V2" \
  --output "$OPD_PROMPTS" \
  --summary-output "$OPD_PROMPTS_SUMMARY" \
  --limit "$OPD_SOURCE_LIMIT" \
  --shuffle \
  --seed 42

python scripts/run_sql_completion_predictions.py \
  --model-path "$TEACHER_MODEL_PATH" \
  --eval-file "$OPD_PROMPTS" \
  --output-file "$TEACHER_PRED" \
  --max-new-tokens 128 \
  --stream-write

python eval/sql_exec_eval.py \
  --predictions "$TEACHER_PRED" \
  --schema-index "$SCHEMA_V2_INDEX" \
  --output "$TEACHER_EVAL" \
  --summary-output "$TEACHER_EVAL_SUMMARY" \
  --timeout-sec 5

python scripts/build_opd_distill_data.py \
  --source-prompts "$OPD_PROMPTS" \
  --teacher-predictions "$TEACHER_PRED" \
  --teacher-exec-eval "$TEACHER_EVAL" \
  --train-output "$OPD_TRAIN" \
  --heldout-output "$OPD_HELDOUT" \
  --summary-output "$OPD_DATA_SUMMARY" \
  --heldout-size "$OPD_HELDOUT_SIZE" \
  --teacher-name "qwen3_8b" \
  --seed 42 \
  --only-teacher-correct

python scripts/train_sql_opd_completion.py \
  --model-path "$MODEL_PATH" \
  --student-adapter-path "$SFT_V3_ADAPTER" \
  --train-file "$OPD_TRAIN" \
  --eval-file "$OPD_HELDOUT" \
  --output-dir "$OPD_OUTPUT" \
  --max-length 1536 \
  --num-train-epochs 1 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-6 \
  --warmup-steps 5 \
  --weight-decay 0.0 \
  --logging-steps 10 \
  --eval-steps 50 \
  --save-steps 50 \
  --save-total-limit 2 \
  --seed 42

python scripts/run_sql_completion_predictions.py \
  --model-path "$MODEL_PATH" \
  --adapter-path "$OPD_OUTPUT" \
  --eval-file "$EVAL_V2" \
  --output-file "$OPD_EVAL_PRED" \
  --max-new-tokens 128 \
  --stream-write

python eval/sql_exec_eval.py \
  --predictions "$OPD_EVAL_PRED" \
  --schema-index "$SCHEMA_V2_INDEX" \
  --output "$OPD_EVAL_OUTPUT" \
  --summary-output "$OPD_EVAL_SUMMARY" \
  --timeout-sec 5

if [[ "$RUN_OPD_RERANK" == "1" ]]; then
  OPD_CANDIDATES="$PROJECT_ROOT/logs/opd_v1_schema_v2_candidates_eval${EVAL_COUNT}.jsonl"
  OPD_RERANK_SELECTED="$PROJECT_ROOT/logs/opd_v1_schema_v2_selected_eval${EVAL_COUNT}.jsonl"
  OPD_RERANK_ANALYSIS="$PROJECT_ROOT/logs/opd_v1_schema_v2_analysis_eval${EVAL_COUNT}.jsonl"
  OPD_RERANK_SUMMARY="$PROJECT_ROOT/logs/opd_v1_schema_v2_summary_eval${EVAL_COUNT}.json"

  python scripts/run_sql_candidate_predictions.py \
    --model-path "$MODEL_PATH" \
    --adapter-path "$OPD_OUTPUT" \
    --eval-file "$EVAL_V2" \
    --output-file "$OPD_CANDIDATES" \
    --max-new-tokens 128 \
    --num-candidates 5 \
    --temperature 0.7 \
    --top-p 0.9

  python scripts/rerank_sql_candidates.py \
    --candidates "$OPD_CANDIDATES" \
    --schema-index "$SCHEMA_V2_INDEX" \
    --sqlite-root "$PROJECT_ROOT/data/raw/spider/database" \
    --selected-output "$OPD_RERANK_SELECTED" \
    --analysis-output "$OPD_RERANK_ANALYSIS" \
    --summary-output "$OPD_RERANK_SUMMARY" \
    --timeout-sec 5
fi

python - "$TEACHER_EVAL_SUMMARY" "$OPD_DATA_SUMMARY" "$OPD_EVAL_SUMMARY" <<'PY'
import json
import sys

teacher = json.load(open(sys.argv[1], encoding="utf-8"))
data = json.load(open(sys.argv[2], encoding="utf-8"))
opd = json.load(open(sys.argv[3], encoding="utf-8"))
print(json.dumps({
    "route": "opd_v1_from_sft_v3_teacher_qwen3_8b",
    "teacher_train_exec_match": teacher.get("exec_match"),
    "teacher_train_total": teacher.get("total"),
    "opd_train_size": data.get("train_size"),
    "opd_heldout_size": data.get("heldout_size_actual"),
    "opd_eval_exec_match": opd.get("exec_match"),
    "opd_eval_total": opd.get("total"),
    "opd_eval_execution_accuracy": opd.get("execution_accuracy"),
    "baseline_sft_v3_schema_v2_prompt_only": "361/500",
    "best_mainline_with_rerank": "404/500",
}, ensure_ascii=False, indent=2))
PY
