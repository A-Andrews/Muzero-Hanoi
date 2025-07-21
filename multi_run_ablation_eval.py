import argparse
import os

import numpy as np
import torch

from acting_experiments.acting_ablations import (
    ablate_networks,
    get_results,
    get_starting_state,
    set_seed,
)
from env.hanoi import TowersOfHanoi
from MCTS.mcts import MCTS
from networks import MuZeroNet


def run_multi(args):
    env = TowersOfHanoi(N=args.N, max_steps=args.max_steps)
    file_indx = get_starting_state(env, args.start)
    s_space_size = env.oneH_s_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_runs = []
    for r in range(args.runs):
        set_seed(args.seed + r)
        mcts = MCTS(
            discount=args.discount,
            root_dirichlet_alpha=args.dirichlet_alpha,
            n_simulations=args.n_mcts_simulations_range[0],
            batch_s=1,
            device=device,
        )
        networks = MuZeroNet(
            rpr_input_s=s_space_size,
            action_s=args.n_action,
            lr=args.lr,
            TD_return=args.TD_return,
            device=device,
        ).to(device)
        ckpt_path = os.path.join("stats", "Hanoi", args.timestamp, "muzero_model.pt")
        ckpt = torch.load(ckpt_path, map_location=device)
        networks.load_state_dict(ckpt["Muzero_net"])
        networks.optimiser.load_state_dict(ckpt["Net_optim"])

        networks = ablate_networks(
            args.reset_latent_policy,
            args.reset_latent_values,
            args.reset_latent_rwds,
            networks,
        )

        data = get_results(
            env,
            args.start,
            networks,
            mcts,
            args.episode,
            args.n_mcts_simulations_range,
            args.temperature,
        )
        all_runs.append([d[1] for d in data])

    arr = np.array(all_runs, dtype=float)  # (runs, len(sim_range))
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    out = np.column_stack([args.n_mcts_simulations_range, mean, std])

    save_dir = os.path.join("stats", "Hanoi", args.timestamp, str(file_indx))
    os.makedirs(save_dir, exist_ok=True)

    label = "".join(
        [
            "ResetLatentPol_" if args.reset_latent_policy else "",
            "ResetLatentVal_" if args.reset_latent_values else "",
            "ResetLatentRwd_" if args.reset_latent_rwds else "",
        ]
    )
    if label == "":
        label = "Muzero_"

    save_path = os.path.join(save_dir, label + "actingAccuracy_error.pt")
    torch.save(torch.tensor(out), save_path)
    print(f"Saved results to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation evaluation multiple times and compute variance"
    )
    parser.add_argument("--timestamp", required=True, help="Model timestamp")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    parser.add_argument("--seed", type=int, default=1, help="Base random seed")
    parser.add_argument("--start", type=int, default=None, help="Start state index")
    parser.add_argument("--reset_latent_policy", action="store_true")
    parser.add_argument("--reset_latent_values", action="store_true")
    parser.add_argument("--reset_latent_rwds", action="store_true")
    parser.add_argument(
        "--n_mcts_simulations_range",
        type=int,
        nargs="+",
        default=[5, 10, 30, 50, 80, 110, 150],
    )
    parser.add_argument("--episode", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--dirichlet_alpha", type=float, default=0)
    parser.add_argument("--discount", type=float, default=0.8)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0)
    parser.add_argument("--TD_return", type=bool, default=True)
    parser.add_argument("--n_action", type=int, default=6)
    args = parser.parse_args()

    run_multi(args)


if __name__ == "__main__":
    main()
