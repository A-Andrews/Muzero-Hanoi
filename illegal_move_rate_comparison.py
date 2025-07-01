import argparse
import copy
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from env.hanoi import TowersOfHanoi
from MCTS.mcts import MCTS
from networks import MuZeroNet
from utils import PLOT_COLORS, set_plot_style, setup_logger


def ablate_networks(networks, ablate_policy=False, ablate_value=False):
    """Return a copy of ``networks`` with selected components reinitialised."""
    ablated = copy.deepcopy(networks)
    if ablate_policy:
        ablated.policy_net.apply(ablated.reset_param)
    if ablate_value:
        ablated.value_net.apply(ablated.reset_param)
    return ablated


def illegal_move_rate(env, networks, mcts, episodes=100, temperature=0.0):
    """Run episodes with the given networks and return mean and std of illegal move rates."""
    episode_rates = []
    for _ in range(episodes):
        state = env.random_reset()
        done = False
        illegal = 0
        moves = 0
        while not done:
            action, _, _ = mcts.run_mcts(
                state, networks, temperature=temperature, deterministic=False
            )
            state, _, done, illegal_move = env.step(action)
            illegal += int(illegal_move)
            moves += 1
        if moves > 0:
            episode_rates.append(illegal / moves)

    if not episode_rates:
        return 0.0, 0.0

    mean_rate = float(np.mean(episode_rates))
    std_rate = float(np.std(episode_rates, ddof=1) / np.sqrt(len(episode_rates)))
    return mean_rate, std_rate


def main(timestamp, episodes):
    set_plot_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TowersOfHanoi(N=3, max_steps=200)
    mcts = MCTS(
        discount=0.8,
        root_dirichlet_alpha=0.25,
        n_simulations=25,
        batch_s=1,
        device=device,
    )

    networks = MuZeroNet(
        rpr_input_s=env.oneH_s_size,
        action_s=6,
        lr=0.0,
        TD_return=True,
        device=device,
    ).to(device)

    model_path = os.path.join("stats", "Hanoi", timestamp, "muzero_model.pt")
    ckpt = torch.load(model_path, map_location=device)
    networks.load_state_dict(ckpt["Muzero_net"])
    networks.optimiser.load_state_dict(ckpt["Net_optim"])
    networks.eval()  # Set model to evaluation mode
    logging.info(f"Loaded model from {model_path}")
    logging.info(f"Evaluating illegal move rates over {episodes} episodes...")

    base_rate, base_std = illegal_move_rate(env, networks, mcts, episodes=episodes)
    logging.info(
        f"Base illegal move rate: {base_rate * 100:.2f}% \u00b1 {base_std * 100:.2f}%"
    )

    pol_net = ablate_networks(networks, ablate_policy=True, ablate_value=False)
    policy_rate, policy_std = illegal_move_rate(env, pol_net, mcts, episodes=episodes)
    logging.info(
        f"Policy ablated illegal move rate: {policy_rate * 100:.2f}% \u00b1 {policy_std * 100:.2f}%"
    )

    val_net = ablate_networks(networks, ablate_policy=False, ablate_value=True)
    value_rate, value_std = illegal_move_rate(env, val_net, mcts, episodes=episodes)
    logging.info(
        f"Value ablated illegal move rate: {value_rate * 100:.2f}% \u00b1 {value_std * 100:.2f}%"
    )

    labels = ["Base", "Policy Ablated", "Value Ablated"]
    rates = [base_rate, policy_rate, value_rate]
    stds = [base_std, policy_std, value_std]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        labels,
        np.array(rates) * 100,
        yerr=np.array(stds) * 100,
        capsize=5,
        color=[PLOT_COLORS[0], PLOT_COLORS[1], PLOT_COLORS[2]],
        edgecolor="none",
    )
    ax.set_ylabel("Illegal move rate")
    ax.set_ylim(0, max(5, max(rates) * 100 + 5))
    for bar, rate, std in zip(bars, rates, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std * 100 + 0.5,
            f"{rate*100:.1f}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    plot_path = os.path.join(
        "stats", "Hanoi", timestamp, f"illegal_move_rate_{timestamp}.png"
    )
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=1200)
    logging.info(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot illegal move rate comparison")
    parser.add_argument(
        "--timestamp",
        type=str,
        required=True,
        help="Timestamp directory containing trained model",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to run for evaluation",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    setup_logger(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args.timestamp, args.episodes)
