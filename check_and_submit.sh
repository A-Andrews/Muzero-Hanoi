#!/bin/bash
# check_and_submit.sh
#
# Checks which analysis jobs are complete for a given timestamp and submits
# the missing ones.  Safe to re-run: will never double-submit a finished job.
#
# Usage:
#   bash check_and_submit.sh --timestamp 1748875208
#   bash check_and_submit.sh --timestamp 1748875208 --model /path/to/model --model_label llama3_8b
#   bash check_and_submit.sh --timestamp 1748875208 --dry_run   # print only, no sbatch
#
# Optional flags:
#   --model         Path to LLM model (required for LLM jobs)
#   --model_label   Short label used in output filenames (required for LLM jobs)
#   --episodes      Episodes per LLM job (default: 50)
#   --noise_scale   Noise std dev for noise sweep (default: 0.5)
#   --layers        Space-separated layer indices to sweep (default: "0 4 8 16 24 31")
#   --dry_run       Print what would be submitted without actually submitting
#   --skip_muzero   Skip MuZero combination checks
#   --skip_llm      Skip all LLM checks

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
TIMESTAMP=""
MODEL=""
MODEL_LABEL=""
EPISODES=50
NOISE_SCALE=0.5
LAYERS=(0 4 8 16 24 31)
DRY_RUN=false
SKIP_MUZERO=false
SKIP_LLM=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --timestamp)   TIMESTAMP="$2";           shift 2 ;;
        --model)       MODEL="$2";               shift 2 ;;
        --model_label) MODEL_LABEL="$2";         shift 2 ;;
        --episodes)    EPISODES="$2";            shift 2 ;;
        --noise_scale) NOISE_SCALE="$2";         shift 2 ;;
        --layers)      IFS=' ' read -r -a LAYERS <<< "$2"; shift 2 ;;
        --dry_run)     DRY_RUN=true;             shift 1 ;;
        --skip_muzero) SKIP_MUZERO=true;         shift 1 ;;
        --skip_llm)    SKIP_LLM=true;            shift 1 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$TIMESTAMP" ]]; then
    echo "Error: --timestamp is required."
    exit 1
fi

STATS_DIR="stats/Hanoi/${TIMESTAMP}"

if [[ ! -d "$STATS_DIR" ]]; then
    echo "Error: stats directory not found: $STATS_DIR"
    exit 1
fi

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
SUBMITTED=0
ALREADY_DONE=0
SKIPPED_NO_MODEL=0

submit() {
    # submit CMD DESCRIPTION
    local cmd="$1"
    local desc="$2"
    echo "  [ SUBMIT ] ${desc}"
    if [[ "$DRY_RUN" == false ]]; then
        eval "$cmd"
    else
        echo "             (dry run) $cmd"
    fi
    (( SUBMITTED++ ))
}

done_msg() {
    echo "  [   OK   ] $1"
    (( ALREADY_DONE++ ))
}

# Check whether a .pt result file exists for a given (directory, label) pair.
# Uses the _actingAccuracy.pt file as the completion signal.
muzero_done() {
    local dir="$1"   # ES / MS / LS
    local label="$2" # e.g. ResetLatentPol_ResetLatentVal_
    [[ -f "${STATS_DIR}/${dir}/${label}actingAccuracy.pt" ]]
}

# Check whether an LLM results JSON exists.
llm_done() {
    local dir="$1"       # ES / MS / LS
    local label="$2"     # model_label
    local prompting="$3" # zero_shot / cot / cot_ablateL8 / etc.
    [[ -f "${STATS_DIR}/${dir}/LLM_${label}_${prompting}_results.json" ]]
}

# --------------------------------------------------------------------------
# NOTE on script ↔ directory mapping (confusingly named):
#   _close.sh  --start 0  →  ES  (Far from goal, 7 moves)
#   _mid.sh    --start 1  →  MS  (Moderate, 3 moves)
#   _far.sh    --start 2  →  LS  (Close to goal, 1 move)
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 1. MuZero combination ablations
# --------------------------------------------------------------------------
if [[ "$SKIP_MUZERO" == false ]]; then
    echo ""
    echo "=== MuZero combination ablations ==="

    declare -A COMBOS
    COMBOS["ResetLatentPol_ResetLatentVal_"]="policy_value"
    COMBOS["ResetLatentPol_ResetLatentRwd_"]="policy_reward"
    COMBOS["ResetLatentPol_ResetLatentVal_ResetLatentRwd_"]="policy_value_reward"

    # dir → script_suffix mapping
    declare -A DIR_SUFFIX
    DIR_SUFFIX["ES"]="close"
    DIR_SUFFIX["MS"]="mid"
    DIR_SUFFIX["LS"]="far"

    for LABEL in "${!COMBOS[@]}"; do
        NAME="${COMBOS[$LABEL]}"
        for DIR in ES MS LS; do
            SUFFIX="${DIR_SUFFIX[$DIR]}"
            SCRIPT="ablation_scripts/run_${NAME}_ablations_${SUFFIX}.sh"
            if muzero_done "$DIR" "$LABEL"; then
                done_msg "MuZero ${NAME} / ${DIR}"
            else
                submit \
                    "sbatch ${SCRIPT} --timestamp ${TIMESTAMP}" \
                    "MuZero ${NAME} / ${DIR}"
            fi
        done
    done
fi

# --------------------------------------------------------------------------
# 2–4. LLM jobs (require --model and --model_label)
# --------------------------------------------------------------------------
if [[ "$SKIP_LLM" == false ]]; then

    if [[ -z "$MODEL" || -z "$MODEL_LABEL" ]]; then
        echo ""
        echo "=== LLM jobs ==="
        echo "  [  SKIP  ] --model and --model_label not provided; skipping all LLM checks."
        echo "             Re-run with --model /path/to/model --model_label <label> to submit LLM jobs."
        (( SKIPPED_NO_MODEL++ ))
    else

        # -----------------------------------------------------------------------
        # 2. LLM baseline (zero_shot + cot for all 3 difficulties)
        # -----------------------------------------------------------------------
        echo ""
        echo "=== LLM baseline (zero_shot + cot, all difficulties) ==="

        declare -A LLM_DIR_START
        LLM_DIR_START["ES"]=0
        LLM_DIR_START["MS"]=1
        LLM_DIR_START["LS"]=2

        for PROMPTING in zero_shot cot; do
            for DIR in ES MS LS; do
                START="${LLM_DIR_START[$DIR]}"
                if llm_done "$DIR" "$MODEL_LABEL" "$PROMPTING"; then
                    done_msg "LLM baseline ${MODEL_LABEL} / ${PROMPTING} / ${DIR}"
                else
                    submit \
                        "sbatch llm_eval/run_llm_eval.sh --timestamp ${TIMESTAMP} --model ${MODEL} --model_label ${MODEL_LABEL} --prompting ${PROMPTING} --start ${START} --episodes ${EPISODES}" \
                        "LLM baseline ${MODEL_LABEL} / ${PROMPTING} / ${DIR}"
                fi
            done
        done

        # -----------------------------------------------------------------------
        # 3. LLM layer ablation sweep (CoT, ES only)
        # -----------------------------------------------------------------------
        echo ""
        echo "=== LLM layer ablation sweep (cot / ES) ==="

        for LAYER in "${LAYERS[@]}"; do
            PROMPTING="cot_ablateL${LAYER}"
            if llm_done "ES" "$MODEL_LABEL" "$PROMPTING"; then
                done_msg "LLM ablation layer ${LAYER} / ES"
            else
                submit \
                    "sbatch --job-name=llm_ablateL${LAYER} llm_eval/run_llm_eval.sh --timestamp ${TIMESTAMP} --model ${MODEL} --model_label ${MODEL_LABEL} --prompting cot --start 0 --episodes ${EPISODES} --ablate_layer ${LAYER}" \
                    "LLM ablation layer ${LAYER} / ES"
            fi
        done

        # -----------------------------------------------------------------------
        # 4. LLM noise injection sweep (CoT, ES only)
        # -----------------------------------------------------------------------
        echo ""
        echo "=== LLM noise injection sweep (cot / ES, scale=${NOISE_SCALE}) ==="

        for LAYER in "${LAYERS[@]}"; do
            PROMPTING="cot_noiseS${NOISE_SCALE}_L${LAYER}"
            if llm_done "ES" "$MODEL_LABEL" "$PROMPTING"; then
                done_msg "LLM noise layer ${LAYER} / ES"
            else
                submit \
                    "sbatch --job-name=llm_noiseL${LAYER} llm_eval/run_llm_eval.sh --timestamp ${TIMESTAMP} --model ${MODEL} --model_label ${MODEL_LABEL} --prompting cot --start 0 --episodes ${EPISODES} --noise_scale ${NOISE_SCALE} --noise_layer ${LAYER}" \
                    "LLM noise layer ${LAYER} / ES"
            fi
        done

    fi
fi

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  Done:      ${ALREADY_DONE}"
echo "  Submitted: ${SUBMITTED}"
[[ $SKIPPED_NO_MODEL -gt 0 ]] && echo "  Skipped (no model path): LLM jobs"
[[ "$DRY_RUN" == true ]]       && echo "  (DRY RUN — nothing actually submitted)"
echo "========================================"
