#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=muzero_hanoi_no_ablations_far
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

echo "Running no ablations far"
python3 acting_experiments/acting_ablations.py --start 2 "$@"

echo "Done!"