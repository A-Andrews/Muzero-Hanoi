#!/bin/bash
# Submit per-difficulty illegal move rate evaluation jobs.
# Runs illegal_move_rate_comparison.py with --start 0/1/2 and --n_simulations 150
# to match the MuZero error evaluation setup.
#
# Usage:
#   bash run_illegal_rate_perdiff.sh --timestamp 1748875208 [--episodes 100] [--n_simulations 150] [--seed 1]

TIMESTAMP=""
EPISODES=100
N_SIMS=150
SEED=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --timestamp) TIMESTAMP="$2"; shift 2;;
        --episodes) EPISODES="$2"; shift 2;;
        --n_simulations) N_SIMS="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [[ -z "$TIMESTAMP" ]]; then
    echo "Error: --timestamp is required"
    exit 1
fi

mkdir -p logs

for START in 0 1 2; do
    case $START in
        0) LABEL="ES";;
        1) LABEL="MS";;
        2) LABEL="LS";;
    esac

    JOB_NAME="illegal_rate_${LABEL}"

    sbatch --partition=short \
           --job-name="$JOB_NAME" \
           --output="logs/${JOB_NAME}_%j.out" \
           --error="logs/${JOB_NAME}_%j.err" \
           --nodes=1 \
           --ntasks-per-node=1 \
           --wrap="
module load Python/3.11.3-GCCcore-12.3.0
source .venv/bin/activate
python3 illegal_move_rate_comparison.py \
    --timestamp $TIMESTAMP \
    --episodes $EPISODES \
    --n_simulations $N_SIMS \
    --seed $SEED \
    --start $START
"
    echo "Submitted: $JOB_NAME (start=$START, $LABEL)"
done
