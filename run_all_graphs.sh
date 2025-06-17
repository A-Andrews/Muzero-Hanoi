#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=run_all_graphs
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

echo "------------------------------------------------"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"
module load Python/3.11.3-GCCcore-12.3.0
source ".venv/bin/activate"

python3 illegal_move_rate_comparison.py "$@"
python3 gradient_analysis.py "$@"
python3 legal_illegal_preds.py "$@"
python3 pre_vs_post_ts.py "$@"
python3 noise_injection_comparison.py "$@"
python3 permutation_importance.py "$@"
python3 acting_experiments/results/plot_startingState_results.py "$@"