#!/bin/bash

sbatch run_multi_ablation.sh "$@"
sbatch run_multi_ablation.sh --start 0 "$@"
sbatch run_multi_ablation.sh --start 1 "$@"
sbatch run_multi_ablation.sh --start 2 "$@"

sbatch run_multi_ablation.sh --reset_latent_policy "$@"
sbatch run_multi_ablation.sh --start 0 --reset_latent_policy "$@"
sbatch run_multi_ablation.sh --start 1 --reset_latent_policy "$@"
sbatch run_multi_ablation.sh --start 2 --reset_latent_policy "$@"

sbatch run_multi_ablation.sh --reset_latent_values "$@"
sbatch run_multi_ablation.sh --start 0 --reset_latent_values "$@"
sbatch run_multi_ablation.sh --start 1 --reset_latent_values "$@"
sbatch run_multi_ablation.sh --start 2 --reset_latent_values "$@"

sbatch run_multi_ablation.sh --reset_latent_rwds "$@"
sbatch run_multi_ablation.sh --start 0 --reset_latent_rwds "$@"
sbatch run_multi_ablation.sh --start 1 --reset_latent_rwds "$@"
sbatch run_multi_ablation.sh --start 2 --reset_latent_rwds "$@"

sbatch run_multi_ablation.sh --reset_latent_values --reset_latent_rwds "$@"
sbatch run_multi_ablation.sh --start 0 --reset_latent_values --reset_latent_rwds "$@"
sbatch run_multi_ablation.sh --start 1 --reset_latent_values --reset_latent_rwds "$@"
sbatch run_multi_ablation.sh --start 2 --reset_latent_values --reset_latent_rwds "$@"