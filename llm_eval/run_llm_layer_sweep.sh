#!/bin/bash
# Submit a layerwise intervention sweep: one SLURM job per layer sample.
#
# Two sweep modes (select with --sweep_type):
#   ablation  — each job ablates one transformer block (skip-layer)
#   noise     — each job injects Gaussian noise at one layer (--noise_scale required)
#
# Usage:
#   bash llm_eval/run_llm_layer_sweep.sh \
#       --timestamp 1748875208 \
#       --model /path/to/Llama-3.1-8B-Instruct \
#       --model_label llama3_8b \
#       --start 0 \
#       --episodes 50 \
#       --sweep_type ablation
#
#   bash llm_eval/run_llm_layer_sweep.sh \
#       --timestamp 1748875208 \
#       --model /path/to/Llama-3.1-8B-Instruct \
#       --model_label llama3_8b \
#       --start 0 \
#       --episodes 50 \
#       --sweep_type noise \
#       --noise_scale 0.5
#
# Default layer sample: [0, 4, 8, 16, 24, 31]  — logarithmic coverage of 32 layers
# Override with --layers "0 4 8 14 20 27" for models with different layer counts

# --------------------------------------------------------------------------
# Parse named arguments
# --------------------------------------------------------------------------
TIMESTAMP=""
MODEL=""
MODEL_LABEL=""
START=0
EPISODES=50
SWEEP_TYPE="ablation"
NOISE_SCALE=0.5
LAYERS_STR=""
# Remaining args forwarded to each individual job (e.g. --prompting cot)
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --timestamp)   TIMESTAMP="$2";   shift 2 ;;
        --model)       MODEL="$2";        shift 2 ;;
        --model_label) MODEL_LABEL="$2";  shift 2 ;;
        --start)       START="$2";        shift 2 ;;
        --episodes)    EPISODES="$2";     shift 2 ;;
        --sweep_type)  SWEEP_TYPE="$2";   shift 2 ;;
        --noise_scale) NOISE_SCALE="$2";  shift 2 ;;
        --layers)      LAYERS_STR="$2";   shift 2 ;;
        *)             EXTRA_ARGS="$EXTRA_ARGS $1 $2"; shift 2 ;;
    esac
done

if [[ -z "$TIMESTAMP" || -z "$MODEL" || -z "$MODEL_LABEL" ]]; then
    echo "Error: --timestamp, --model, and --model_label are required."
    exit 1
fi

# Layer sample — override with --layers for non-32-layer models
if [[ -n "$LAYERS_STR" ]]; then
    read -ra LAYERS <<< "$LAYERS_STR"
else
    LAYERS=(0 4 8 16 24 31)
fi

echo "Submitting ${SWEEP_TYPE} sweep over layers: ${LAYERS[*]}"
echo "  timestamp=${TIMESTAMP}  model_label=${MODEL_LABEL}  start=${START}  episodes=${EPISODES}"

for LAYER in "${LAYERS[@]}"; do
    if [[ "$SWEEP_TYPE" == "ablation" ]]; then
        INTERVENTION_ARGS="--ablate_layer ${LAYER}"
        JOB_NAME="llm_ablateL${LAYER}"
    elif [[ "$SWEEP_TYPE" == "noise" ]]; then
        INTERVENTION_ARGS="--noise_scale ${NOISE_SCALE} --noise_layer ${LAYER}"
        JOB_NAME="llm_noiseL${LAYER}"
    else
        echo "Unknown sweep_type: ${SWEEP_TYPE}. Use 'ablation' or 'noise'."
        exit 1
    fi

    sbatch --job-name="${JOB_NAME}" \
        llm_eval/run_llm_eval.sh \
        --timestamp "${TIMESTAMP}" \
        --model "${MODEL}" \
        --model_label "${MODEL_LABEL}" \
        --prompting cot \
        --start "${START}" \
        --episodes "${EPISODES}" \
        ${INTERVENTION_ARGS} \
        ${EXTRA_ARGS}

    echo "  Submitted layer=${LAYER}  (${SWEEP_TYPE})"
done

echo ""
echo "All ${#LAYERS[@]} jobs submitted."
echo "Also submit a baseline (no intervention) if not already done:"
echo "  sbatch llm_eval/run_llm_eval.sh --timestamp ${TIMESTAMP} --model ${MODEL} --model_label ${MODEL_LABEL} --prompting cot --start ${START} --episodes ${EPISODES}"
