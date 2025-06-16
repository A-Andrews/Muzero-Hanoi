import argparse
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from env.hanoi import TowersOfHanoi
from MCTS.mcts import MCTS
from networks import MuZeroNet
from utils import PLOT_COLORS, oneHot_encoding, set_plot_style, setup_logger


def play_game(networks, env, start_state, max_game_steps, mcts, noise_std=0.0):
    """Play one game of Towers of Hanoi with optional input noise."""
    env.reset()
    env.c_state = start_state
    env.oneH_c_state = oneHot_encoding(env.c_state, n_integers=env.n_pegs)

    for _ in range(max_game_steps):
        state_one_hot = oneHot_encoding(env.c_state, n_integers=env.n_pegs)
        if noise_std > 0:
            noise = np.random.normal(0.0, noise_std, size=state_one_hot.shape)
            state_one_hot = state_one_hot + noise

        action, _, _ = mcts.run_mcts(
            state_one_hot, networks, temperature=0, deterministic=True
        )
        _, reward, done, _ = env.step(action)
        if reward == 100:
            return True
        if done:
            return False
    return False


def run_evaluation(networks, env, test_states, max_game_steps, mcts, noise_std=0.0):
    """Evaluate solve rate of ``networks`` over ``test_states`` with noise."""
    solved = 0
    for start_state in tqdm(test_states, desc=f"Noise {noise_std}"):
        if play_game(networks, env, start_state, max_game_steps, mcts, noise_std):
            solved += 1
    return solved / len(test_states)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate how injected noise affects solve rate"
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        required=True,
        help="Timestamp of the trained model directory",
    )
    parser.add_argument("--N", type=int, default=3, help="Number of disks")
    parser.add_argument(
        "--noise_stds",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        help="Space separated list of noise std values to evaluate",
    )
    parser.add_argument(
        "--num_states",
        type=int,
        default=100,
        help="Number of random start states for evaluation",
    )
    parser.add_argument("--max_game_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    setup_logger(args.seed)
    set_plot_style()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TowersOfHanoi(N=args.N, max_steps=args.max_game_steps)

    mcts = MCTS(
        n_simulations=25,
        root_dirichlet_alpha=0.25,
        discount=0.8,
        batch_s=1,
        device=device,
    )

    networks = MuZeroNet(
        action_s=6,
        lr=0.002,
        rpr_input_s=env.oneH_s_size,
        device=device,
        TD_return=True,
    )
    model_path = os.path.join("stats", "Hanoi", args.timestamp, "muzero_model.pt")
    ckpt = torch.load(model_path, map_location=device)
    networks.load_state_dict(ckpt["Muzero_net"])
    networks.eval()

    all_states = [s for s in env.states if s != env.goal]
    if args.num_states <= len(all_states):
        test_states = random.sample(all_states, args.num_states)
    else:
        test_states = [random.choice(all_states) for _ in range(args.num_states)]

    logging.info(f"Using a test set of {len(test_states)} starting states.")

    solve_rates = []
    for std in args.noise_stds:
        rate = run_evaluation(
            networks, env, test_states, args.max_game_steps, mcts, noise_std=std
        )
        solve_rates.append(rate)
        if std == 0:
            logging.info(f"Baseline solve rate: {rate:.2%}")
        else:
            logging.info(
                f"Noise std {std:.2f} -> solve rate {rate:.2%} (diff {solve_rates[0] - rate:.2%})"
            )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(args.noise_stds, solve_rates, marker="o", color=PLOT_COLORS[0])
    ax.set_xlabel("Noise Std Dev")
    ax.set_ylabel("Solve Rate")
    ax.set_title("Performance vs Injected Noise")
    ax.set_ylim(0, 1)
    ax.grid(True)
    fig.tight_layout()

    save_dir = os.path.join("stats", "Hanoi", args.timestamp)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "noise_vs_performance.png")
    fig.savefig(save_path)
    logging.info(f"Plot saved to {save_path}")


if __name__ == "__main__":
    main()
