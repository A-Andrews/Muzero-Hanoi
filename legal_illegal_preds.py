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
from utils import setup_logger


def ablate_network(networks, ablate_policy=False, ablate_value=False):
    """Return a copy of ``networks`` with selected components reset."""
    net_copy = copy.deepcopy(networks)
    if ablate_policy:
        net_copy.policy_net.apply(net_copy.reset_param)
    if ablate_value:
        net_copy.value_net.apply(net_copy.reset_param)
    return net_copy


def evaluate_network(env, mcts, networks, episodes, device):
    """Collect predicted rewards and values for legal and illegal moves."""
    reward_diffs = []
    value_diffs = []
    for _ in range(episodes):
        state = env.random_reset()
        done = False
        legal_rewards, illegal_rewards = [], []
        legal_values, illegal_values = [], []
        while not done:
            action, _, _ = mcts.run_mcts(
                state, networks, temperature=0.0, deterministic=False
            )
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(
                0
            )
            h_state = networks.represent(state_t)
            action_t = torch.tensor([action], dtype=torch.long, device=device)
            action_oh = torch.nn.functional.one_hot(
                action_t, num_classes=networks.num_actions
            ).float()
            _, pred_reward, _, pred_value = networks.recurrent_inference(
                h_state, action_oh
            )
            next_state, _, done, illegal = env.step(action)
            if illegal:
                illegal_rewards.append(pred_reward)
                illegal_values.append(pred_value)
            else:
                legal_rewards.append(pred_reward)
                legal_values.append(pred_value)
            state = next_state
        legal_reward = np.mean(legal_rewards) if legal_rewards else 0.0
        illegal_reward = np.mean(illegal_rewards) if illegal_rewards else 0.0
        legal_value = np.mean(legal_values) if legal_values else 0.0
        illegal_value = np.mean(illegal_values) if illegal_values else 0.0

        reward_diffs.append(legal_reward - illegal_reward)
        value_diffs.append(legal_value - illegal_value)

    return {
        "reward_diff_mean": float(np.mean(reward_diffs)) if reward_diffs else 0.0,
        "reward_diff_std": float(np.std(reward_diffs)) if reward_diffs else 0.0,
        "value_diff_mean": float(np.mean(value_diffs)) if value_diffs else 0.0,
        "value_diff_std": float(np.std(value_diffs)) if value_diffs else 0.0,
    }


def plot_results(results, save_path):
    labels = list(results.keys())
    reward_means = [results[l]["reward_diff_mean"] for l in labels]
    reward_stds = [results[l]["reward_diff_std"] for l in labels]
    value_means = [results[l]["value_diff_mean"] for l in labels]
    value_stds = [results[l]["value_diff_std"] for l in labels]

    x = np.arange(len(labels))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    bars1 = axs[0].bar(
        x,
        reward_means,
        yerr=reward_stds,
        capsize=5,
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
        edgecolor="black",
    )
    axs[0].set_ylabel("Predicted Reward Difference")
    axs[0].set_title("Legal - Illegal")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels, rotation=45)
    for bar, val in zip(bars1, reward_means):
        axs[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
        )

    bars2 = axs[1].bar(
        x,
        value_means,
        yerr=value_stds,
        capsize=5,
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
        edgecolor="black",
    )
    axs[1].set_ylabel("Predicted Value Difference")
    axs[1].set_title("Legal - Illegal")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels, rotation=45)
    for bar, val in zip(bars2, value_means):
        axs[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    logging.info(f"Saved plot to {save_path}")


def main(timestamp, episodes):
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
    networks.eval()

    exp_networks = {
        "Base": networks,
        "Policy Ablated": ablate_network(networks, ablate_policy=True),
        "Value Ablated": ablate_network(networks, ablate_value=True),
    }

    results = {}
    for label, net in exp_networks.items():
        net.eval()
        logging.info(f"Evaluating {label} network...")
        results[label] = evaluate_network(env, mcts, net, episodes, device)

    plot_path = os.path.join("stats", "Hanoi", timestamp, "reward_value_legality.png")
    plot_results(results, plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot reward and value change for legal vs illegal moves"
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        required=True,
        help="Timestamp directory containing trained model",
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes to run"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    setup_logger(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args.timestamp, args.episodes)
