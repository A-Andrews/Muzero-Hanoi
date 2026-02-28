#!/bin/bash
# Submit LLM evaluation jobs for all 6 conditions:
#   3 difficulties (Close=LS, Moderate=MS, Far=ES)  ×  2 prompting strategies
#
# Usage:
#   cd /well/costa/users/zqa082/Muzero-Hanoi
#   bash llm_eval/run_all_llm_eval.sh \
#       --timestamp 1748875208 \
#       --model /well/costa/users/zqa082/models/Llama-3.1-8B-Instruct \
#       --model_label llama3_8b \
#       --episodes 50
#
# All remaining flags are forwarded to llm_hanoi_eval.py, so you can also
# pass --temperature, --max_new_tokens, --seed, --dtype, etc.

set -euo pipefail

# ---- parse our own flags so we know the model_label ----
TIMESTAMP=""
MODEL_LABEL="llm"
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

# Difficulties: start=0 → Far (ES), start=1 → Moderate (MS), start=2 → Close (LS)
DIFFICULTIES=(0 1 2)
PROMPTING=("zero_shot" "cot")

for START in "${DIFFICULTIES[@]}"; do
    for PROM in "${PROMPTING[@]}"; do
        JOB_NAME="llm_${MODEL_LABEL}_s${START}_${PROM}"
        echo "Submitting: ${JOB_NAME}"
        sbatch \
            --job-name="${JOB_NAME}" \
            llm_eval/run_llm_eval.sh \
            "${POSITIONAL_ARGS[@]}" \
            --prompting "${PROM}" \
            --start "${START}"
    done
done

echo ""
echo "All 6 jobs submitted. Monitor with: squeue -u \$(whoami)"
echo ""
echo "Once all jobs finish, generate comparison plots with:"
echo "  python3 llm_eval/plot_llm_comparison.py \\"
echo "      --timestamp ${TIMESTAMP} \\"
echo "      --model_label ${MODEL_LABEL} \\"
echo "      --llm_display_names \"${MODEL_LABEL} (zero-shot)\" \"${MODEL_LABEL} (CoT)\""
