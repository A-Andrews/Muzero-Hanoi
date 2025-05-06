#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=hello_world_python
#SBATCH --output=job_outputs/%x_%j.out
#SBATCH --error=job_outputs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

echo "------------------------------------------------"
echo "Run on host: "`hostname`
echo "Operating system: "`uname -s`
echo "Username: "`whoami`
echo "Started at: "`date`
echo "------------------------------------------------"
sleep 60s
module load Python/3.11.3-GCCcore-12.3.0
python3 hello.py