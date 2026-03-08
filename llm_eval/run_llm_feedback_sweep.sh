#!/bin/bash
# Submit LLM evaluation jobs for history + feedback conditions.
#
# Conditions submitted (per difficulty × strategy):
#   h{N}          — history_length=N, no feedback
#   h{N}_illfb    — history_length=N, illegal feedback enabled
#
# Baselines (h=0) are NOT re-submitted — use run_all_llm_eval.sh for those.
#
# Usage:
#   bash llm_eval/run_llm_feedback_sweep.sh \
#       --timestamp 1748875208 \
#       --model /path/to/model \
#       --model_label qwen25_7b \
#       --episodes 50 \
#       --history_length 5
#
# Extra flags (--temperature, --seed, etc.) are forwarded to llm_hanoi_eval.py.

set -euo pipefail

TIMESTAMP=""
MODEL_LABEL="llm"
HISTORY_LENGTH=5
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --timestamp)
            TIMESTAMP="$2"
            POSITIONAL_ARGS+=("$1" "$2")
            shift 2 ;;
        --model_label)
            MODEL_LABEL="$2"
            POSITIONAL_ARGS+=("$1" "$2")
            shift 2 ;;
        --history_length)
            HISTORY_LENGTH="$2"
            # Do NOT add to POSITIONAL_ARGS — we add it per-job below
            shift 2 ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift ;;
    esac
done

if [[ -z "$TIMESTAMP" ]]; then
    echo "ERROR: --timestamp is required" >&2
    exit 1
fi

mkdir -p logs

DIFFICULTIES=(0 1 2)
PROMPTING=("cot")   # history is most meaningful with CoT

echo "=== History/Feedback sweep ==="
echo "  model_label=${MODEL_LABEL}  history_length=${HISTORY_LENGTH}"
echo ""

for START in "${DIFFICULTIES[@]}"; do
    for PROM in "${PROMPTING[@]}"; do
        # Condition 1: history only (no feedback)
        JOB_NAME="llm_${MODEL_LABEL}_s${START}_${PROM}_h${HISTORY_LENGTH}"
        echo "Submitting: ${JOB_NAME}"
        sbatch --job-name="${JOB_NAME}" \
            llm_eval/run_llm_eval.sh \
            "${POSITIONAL_ARGS[@]}" \
            --prompting "${PROM}" \
            --start "${START}" \
            --history_length "${HISTORY_LENGTH}"

        # Condition 2: history + illegal feedback
        JOB_NAME="llm_${MODEL_LABEL}_s${START}_${PROM}_h${HISTORY_LENGTH}_illfb"
        echo "Submitting: ${JOB_NAME}"
        sbatch --job-name="${JOB_NAME}" \
            llm_eval/run_llm_eval.sh \
            "${POSITIONAL_ARGS[@]}" \
            --prompting "${PROM}" \
            --start "${START}" \
            --history_length "${HISTORY_LENGTH}" \
            --illegal_feedback
    done
done

echo ""
echo "Submitted $(( ${#DIFFICULTIES[@]} * ${#PROMPTING[@]} * 2 )) jobs."
echo "Monitor with: squeue -u \$(whoami)"
echo ""
echo "To plot, pass the new prompting strategies to plot_llm_comparison.py:"
echo "  --prompting_strategies cot cot_h${HISTORY_LENGTH} cot_h${HISTORY_LENGTH}_illfb"
