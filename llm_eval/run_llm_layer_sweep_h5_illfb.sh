#!/bin/bash
# Submit layerwise intervention sweeps WITH history + illegal feedback.
#
# Re-runs the ablation and noise sweeps using --history_length 5 --illegal_feedback
# so results are methodologically valid (avoids degenerate illegal-move loops).
#
# Results are saved separately from the original runs because the prompting label
# includes "_h5_illfb" (e.g. LLM_qwen25_7b_cot_ablateL4_h5_illfb_actingAccuracy.pt).
#
# Also submits baseline (no intervention) h5_illfb jobs per difficulty if needed.
#
# Usage:
#   bash llm_eval/run_llm_layer_sweep_h5_illfb.sh

set -euo pipefail

TIMESTAMP="1748875208"
MODEL="/well/costa/users/zqa082/models/Qwen2.5-7B-Instruct"
MODEL_LABEL="qwen25_7b"
EPISODES=50
LAYERS="0 4 8 14 20 27"
NOISE_SCALE=0.5

mkdir -p logs

echo "=== Layer sweep with h5_illfb ==="
echo "  model=${MODEL}"
echo "  layers=${LAYERS}"
echo "  episodes=${EPISODES}"
echo ""

for START in 0 1 2; do
    case $START in
        0) DIFF="ES (Far)" ;;
        1) DIFF="MS (Moderate)" ;;
        2) DIFF="LS (Close)" ;;
    esac

    echo "--- Difficulty: ${DIFF} (start=${START}) ---"

    # Baseline: cot + h5_illfb, no intervention
    JOB_NAME="llm_${MODEL_LABEL}_s${START}_cot_h5illfb_baseline"
    echo "  Submitting baseline: ${JOB_NAME}"
    sbatch --job-name="${JOB_NAME}" \
        llm_eval/run_llm_eval.sh \
        --timestamp "${TIMESTAMP}" \
        --model "${MODEL}" \
        --model_label "${MODEL_LABEL}" \
        --prompting cot \
        --start "${START}" \
        --episodes "${EPISODES}" \
        --history_length 5 \
        --illegal_feedback

    # Ablation sweep
    bash llm_eval/run_llm_layer_sweep.sh \
        --timestamp "${TIMESTAMP}" \
        --model "${MODEL}" \
        --model_label "${MODEL_LABEL}" \
        --start "${START}" \
        --episodes "${EPISODES}" \
        --sweep_type ablation \
        --layers ${LAYERS} \
        --history_length 5 --illegal_feedback

    # Noise sweep
    bash llm_eval/run_llm_layer_sweep.sh \
        --timestamp "${TIMESTAMP}" \
        --model "${MODEL}" \
        --model_label "${MODEL_LABEL}" \
        --start "${START}" \
        --episodes "${EPISODES}" \
        --sweep_type noise \
        --noise_scale "${NOISE_SCALE}" \
        --layers ${LAYERS} \
        --history_length 5 --illegal_feedback

    echo ""
done

echo "=== All jobs submitted ==="
echo "  3 baselines + 3×6 ablation + 3×6 noise = 39 jobs total"
echo "  Monitor with: squeue -u \$(whoami)"
echo ""
echo "Results will be saved with '_h5_illfb' suffix, separate from original runs."
