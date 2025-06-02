#!/bin/bash
#SBATCH --partition=gpu_long
#SBATCH --job-name=muzero_hanoi_value_ablations_rand
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

echo "------------------------------------------------"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"
module load Python/3.11.3-GCCcore-12.3.0
source ".venv/bin/activate"

echo "Running ablations to value"
python3 acting_experiments/acting_ablations.py --reset_latent_values True "$@"

echo "Done!"