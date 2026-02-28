#!/bin/bash
# SLURM batch script for a SINGLE LLM evaluation condition.
#
# Submit via run_all_llm_eval.sh, or directly:
#   sbatch llm_eval/run_llm_eval.sh \
#       --timestamp 1748875208 \
#       --model /path/to/model \
#       --model_label llama3_8b \
#       --prompting zero_shot \
#       --start 0 \
#       --episodes 50
#
#SBATCH --job-name=llm_hanoi_eval
#SBATCH --partition=gpu_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "------------------------------------------------"
echo "Run on host: $(hostname)"
echo "Operating system: $(uname -s)"
echo "Username: $(whoami)"
echo "Started at: $(date)"
echo "SLURM job ID: ${SLURM_JOB_ID}"
echo "------------------------------------------------"

# libffi.so.8 must be on LD_LIBRARY_PATH before Python loads ctypes.
# The path below is the known location on this cluster (skylake, GCCcore-12.3.0).
export LD_LIBRARY_PATH="/apps/eb/el8/2023a/skylake/software/libffi/3.4.4-GCCcore-12.3.0/lib64:${LD_LIBRARY_PATH:-}"

module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

source ".venv/bin/activate"

# Optional: point HuggingFace cache at a shared location on /well so weights
# are not re-downloaded for every job
export HF_HOME="/well/costa/users/zqa082/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

echo "Python: $(which python3)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo ""

python3 llm_eval/llm_hanoi_eval.py "$@"

echo "------------------------------------------------"
echo "Finished at: $(date)"
echo "------------------------------------------------"
