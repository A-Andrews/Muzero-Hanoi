#!/bin/bash
# Submit targeted Llama-3.1-8B-Instruct evaluation jobs.
#
# Runs the key subset of conditions to confirm Qwen findings generalise:
#   - zero_shot × 3 difficulties
#   - cot × 3 difficulties
#   - cot_h5_illfb × 3 difficulties  (best feedback condition)
#   - cot + ablate layer 0 × 3 difficulties  (critical layer)
#   - cot + ablate layer 14 × 3 difficulties (non-critical layer)
#
# Total: 15 jobs
#
# Usage:
#   bash llm_eval/run_llama_targeted.sh

set -euo pipefail

TIMESTAMP="1748875208"
MODEL="/well/costa/users/zqa082/models/Llama-3.1-8B-Instruct"
MODEL_LABEL="llama3_8b"
EPISODES=50

mkdir -p logs

echo "=== Targeted Llama-3.1-8B-Instruct evaluation ==="
echo "  timestamp=${TIMESTAMP}  episodes=${EPISODES}"
echo ""

COUNT=0

for START in 0 1 2; do
    # 1. Zero-shot baseline
    JOB="llama_s${START}_zs"
    echo "Submitting: ${JOB}"
    sbatch --job-name="${JOB}" \
        llm_eval/run_llm_eval.sh \
        --timestamp "${TIMESTAMP}" \
        --model "${MODEL}" \
        --model_label "${MODEL_LABEL}" \
        --prompting zero_shot \
        --start "${START}" \
        --episodes "${EPISODES}"
    COUNT=$((COUNT + 1))

    # 2. CoT baseline
    JOB="llama_s${START}_cot"
    echo "Submitting: ${JOB}"
    sbatch --job-name="${JOB}" \
        llm_eval/run_llm_eval.sh \
        --timestamp "${TIMESTAMP}" \
        --model "${MODEL}" \
        --model_label "${MODEL_LABEL}" \
        --prompting cot \
        --start "${START}" \
        --episodes "${EPISODES}"
    COUNT=$((COUNT + 1))

    # 3. CoT + horizon 5 + illegal feedback
    JOB="llama_s${START}_cot_h5_illfb"
    echo "Submitting: ${JOB}"
    sbatch --job-name="${JOB}" \
        llm_eval/run_llm_eval.sh \
        --timestamp "${TIMESTAMP}" \
        --model "${MODEL}" \
        --model_label "${MODEL_LABEL}" \
        --prompting cot \
        --start "${START}" \
        --episodes "${EPISODES}" \
        --history_length 5 \
        --illegal_feedback
    COUNT=$((COUNT + 1))

    # 4. CoT + ablate layer 0 (critical)
    JOB="llama_s${START}_ablateL0"
    echo "Submitting: ${JOB}"
    sbatch --job-name="${JOB}" \
        llm_eval/run_llm_eval.sh \
        --timestamp "${TIMESTAMP}" \
        --model "${MODEL}" \
        --model_label "${MODEL_LABEL}" \
        --prompting cot \
        --start "${START}" \
        --episodes "${EPISODES}" \
        --ablate_layer 0
    COUNT=$((COUNT + 1))

    # 5. CoT + ablate layer 14 (non-critical)
    JOB="llama_s${START}_ablateL14"
    echo "Submitting: ${JOB}"
    sbatch --job-name="${JOB}" \
        llm_eval/run_llm_eval.sh \
        --timestamp "${TIMESTAMP}" \
        --model "${MODEL}" \
        --model_label "${MODEL_LABEL}" \
        --prompting cot \
        --start "${START}" \
        --episodes "${EPISODES}" \
        --ablate_layer 14
    COUNT=$((COUNT + 1))
done

echo ""
echo "Submitted ${COUNT} jobs."
echo "Monitor with: squeue -u $(whoami)"
echo ""
echo "Results will be saved to: stats/Hanoi/${TIMESTAMP}/{ES,MS,LS}/LLM_${MODEL_LABEL}_*.json"
