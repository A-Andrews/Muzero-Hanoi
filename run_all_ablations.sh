#!/bin/bash
#SBATCH --partition=gpu_long
#SBATCH --job-name=muzero_hanoi_all_ablation_results
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

echo "Running no ablations"
python3 acting_experiments/acting_ablations.py --start 0
python3 acting_experiments/acting_ablations.py --start 1
python3 acting_experiments/acting_ablations.py --start 2

echo "Running ablations to policy"
python3 acting_experiments/acting_ablations.py --start 0 --reset_latent_policy True
python3 acting_experiments/acting_ablations.py --start 1 --reset_latent_policy True
python3 acting_experiments/acting_ablations.py --start 2 --reset_latent_policy True

echo "Running ablations to value"
python3 acting_experiments/acting_ablations.py --start 0 --reset_latent_values True
python3 acting_experiments/acting_ablations.py --start 1 --reset_latent_values True
python3 acting_experiments/acting_ablations.py --start 2 --reset_latent_values True

echo "Running ablations to reward"
python3 acting_experiments/acting_ablations.py --start 0 --reset_latent_rwds True
python3 acting_experiments/acting_ablations.py --start 1 --reset_latent_rwds True
python3 acting_experiments/acting_ablations.py --start 2 --reset_latent_rwds True

echo "Running ablations to value and reward"
python3 acting_experiments/acting_ablations.py --start 0 --reset_latent_values True --reset_latent_rwds True
python3 acting_experiments/acting_ablations.py --start 1 --reset_latent_values True --reset_latent_rwds True
python3 acting_experiments/acting_ablations.py --start 2 --reset_latent_values True --reset_latent_rwds True

echo "Done!"