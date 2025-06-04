#!/bin/bash
#SBATCH --partition=gpu_short
#SBATCH --job-name=gradcam_analysis
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
source .venv/bin/activate

# Run Grad-CAM analysis
python3 gradcam_analysis.py \
    --model_path "results/Hanoi/1/muzero_model.pt" \
    --save_dir "gradcam_results" \
    --layer "representation_net.0"

echo "Grad-CAM analysis completed!"