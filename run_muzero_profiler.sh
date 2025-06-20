#!/bin/bash
#SBATCH --partition=gpu_short
#SBATCH --job-name=muzero_profiler
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
sleep 60s
module load Python/3.11.3-GCCcore-12.3.0
source ".venv/bin/activate"
python3 training_main.py \
    --n_ep_x_loop 5 \
    --n_update_x_loop 5 \
    --batch_s 64 \
    --discount 0.7 \
    --training_loops 10 \
    --profile True \
    --buffer_size 10000