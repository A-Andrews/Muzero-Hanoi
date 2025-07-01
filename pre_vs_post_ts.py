import argparse
import copy
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import setup_logger

# Add repository root to Python path for absolute imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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


def load_network(timestamp, device, env):
    """Load a trained MuZero network from the stats directory."""
    network = MuZeroNet(
        rpr_input_s=env.oneH_s_size,
        action_s=len(env.moves),
        lr=0,
        TD_return=True,
        device=device,
    ).to(device)
    model_path = os.path.join("stats", "Hanoi", timestamp, "muzero_model.pt")
    model_dict = torch.load(model_path, map_location=device)
    network.load_state_dict(model_dict["Muzero_net"])
    network.eval()
    return network


def get_policies(network, mcts, state):
    """Return initial policy from the network and the policy after MCTS."""
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    _, _, init_policy, _ = network.initial_inference(state_t)
    _, final_policy, _ = mcts.run_mcts(
        state, network, temperature=0, deterministic=True
    )
    return init_policy, final_policy


def main():
    parser = argparse.ArgumentParser(description="Compare initial and MCTS policies")
    parser.add_argument("--timestamp", required=True, help="Timestamp of saved model")
    parser.add_argument(
        "--n_simulations", type=int, default=25, help="MCTS simulations"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    setup_logger(args.seed)
    set_plot_style()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TowersOfHanoi(N=3, max_steps=200)

    mcts = MCTS(
        discount=0.8,
        root_dirichlet_alpha=0.0,
        n_simulations=args.n_simulations,
        batch_s=1,
        device=device,
    )

    base_net = load_network(args.timestamp, device, env)

    policy_ab_net = ablate_networks(base_net, ablate_policy=True, ablate_value=False)

    value_ab_net = ablate_networks(base_net, ablate_policy=False, ablate_value=True)

    networks = {
        "Base": base_net,
        "Policy Ablated": policy_ab_net,
        "Value Ablated": value_ab_net,
    }

    state = env.reset()

    policies = {}
    for name, net in networks.items():
        init_pol, final_pol = get_policies(net, mcts, state)
        policies[name] = (init_pol, final_pol)

    action_labels = [f"{a[0]}\u2192{a[1]}" for a in env.moves]
    x = np.arange(len(action_labels))

    fig, axs = plt.subplots(1, len(policies), figsize=(12, 4), sharey=True)
    if len(policies) == 1:
        axs = [axs]
    for ax, (name, (init_pol, final_pol)) in zip(axs, policies.items()):
        ax.bar(x - 0.2, init_pol, width=0.4, label="Initial")
        ax.bar(x + 0.2, final_pol, width=0.4, label="After MCTS")
        ax.set_title(name)
        ax.set_xticks(x)
        ax.set_xticklabels(action_labels, rotation=45)
        ax.set_ylim(0, 1)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    axs[0].set_ylabel("Probability")
    axs[-1].legend(loc="upper right")
    plt.tight_layout()

    save_dir = os.path.join("stats", "Hanoi", args.timestamp)
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, "policy_evolution.png")
    plt.savefig(fig_path, dpi=300)
    print(f"Plot saved to {fig_path}")


if __name__ == "__main__":
    main()
