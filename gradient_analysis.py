import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from env.hanoi import TowersOfHanoi
from MCTS.mcts import MCTS
from networks import MuZeroNet
from utils import oneHot_encoding, setup_logger

# import model
# run model with incorrect state
# record gradients of networks


def save_results(data, timestamp, state):
    state_name = "_".join(map(str, state))
    file_dir = os.path.join("stats", "Hanoi", timestamp, "gradients", state_name)
    os.makedirs(file_dir, exist_ok=True)

    for network, gradients in data.items():
        file_path = os.path.join(file_dir, f"{network}_gradients.pt")
        torch.save(gradients, file_path)
        logging.info(f"Saved gradients for {network} to {file_path}")


def visualize_gradients_subgraphs(gradients_dict, timestamp, state):
    state_name = "_".join(map(str, state))
    file_dir = os.path.join("stats", "Hanoi", timestamp, "gradients", state_name)
    os.makedirs(file_dir, exist_ok=True)
    num_nets = len(gradients_dict)
    net_names = list(gradients_dict.keys())

    fig, axes = plt.subplots(
        nrows=num_nets,
        ncols=3,
        figsize=(18, 5 * num_nets),
        gridspec_kw={"width_ratios": [1, 1, 1]},
    )

    if num_nets == 1:
        axes = axes[np.newaxis, :]  # To always index as [net_idx, col]

    for net_idx, net_name in enumerate(net_names):
        grads = [
            g
            for g in gradients_dict[net_name]
            if g is not None and isinstance(g, torch.Tensor)
        ]
        # 1. Mean Absolute Gradient per Layer (Bar)
        magnitudes = [g.abs().mean().item() for g in grads]
        layers = np.arange(len(magnitudes))
        axes[net_idx, 0].bar(layers, magnitudes, color="skyblue")
        axes[net_idx, 0].set_title(f"{net_name.capitalize()} - Mean Abs Grad")
        axes[net_idx, 0].set_xlabel("Layer")
        axes[net_idx, 0].set_ylabel("Mean |Grad|")

        # 2. Gradient Distribution (Histogram)
        if grads:
            all_grads = torch.cat([g.flatten() for g in grads])
            axes[net_idx, 1].hist(
                all_grads.detach().cpu().numpy(), bins=80, color="salmon", alpha=0.7
            )
            axes[net_idx, 1].set_title(f"{net_name.capitalize()} - Grad Distribution")
            axes[net_idx, 1].set_xlabel("Gradient Value")
            axes[net_idx, 1].set_ylabel("Frequency")
        else:
            axes[net_idx, 1].text(0.5, 0.5, "No gradients", ha="center", va="center")

        # 3. Gradient Norm per Layer (Line)
        norms = [g.norm().item() for g in grads]
        axes[net_idx, 2].plot(layers, norms, "-o", color="seagreen")
        axes[net_idx, 2].set_title(f"{net_name.capitalize()} - Grad Norms")
        axes[net_idx, 2].set_xlabel("Layer")
        axes[net_idx, 2].set_ylabel("||Grad|| (L2)")

    plt.tight_layout()
    plt.savefig(
        os.path.join(file_dir, f"gradients_subgraphs.png"),
        dpi=300,
        bbox_inches="tight",
        format="png",
    )


def get_results(env, networks, mcts, state):
    """
    Run the model with a given state and collect results.
    """
    data = {}
    # Set an initial state
    oneH_c_state = torch.tensor(
        oneHot_encoding(state, n_integers=N), dtype=torch.float32
    ).unsqueeze(0)

    # Representation
    h_state = networks.represent(oneH_c_state)

    # Prediction
    pi_logits, value = networks.prediction(h_state)

    pi_probs = F.softmax(pi_logits, dim=-1)
    rwd = torch.zeros_like(value)  # Initial inference has zero reward

    dummy_action = torch.zeros(1, networks.num_actions, device=networks.dev)
    dummy_action[0, 0] = 1.0  # Set first action to 1

    # Use dynamics network
    next_h_state, next_rwd = networks.dynamics(h_state, dummy_action)

    # get gradients of each network
    representation_grad = torch.autograd.grad(
        outputs=h_state.mean(),
        inputs=networks.representation_net.parameters(),
        retain_graph=True,
        create_graph=True,
    )
    dynamic_grad = torch.autograd.grad(
        outputs=next_h_state.mean(),
        inputs=networks.dynamic_net.parameters(),
        retain_graph=True,
        create_graph=True,
    )
    rwd_grad = torch.autograd.grad(
        outputs=next_rwd.mean(),
        inputs=networks.rwd_net.parameters(),
        retain_graph=True,
        create_graph=True,
    )
    policy_grad = torch.autograd.grad(
        outputs=pi_probs.mean(),
        inputs=networks.policy_net.parameters(),
        retain_graph=True,
        create_graph=True,
    )
    value_grad = torch.autograd.grad(
        outputs=value.mean(),
        inputs=networks.value_net.parameters(),
        retain_graph=True,
        create_graph=True,
    )
    # add to dictionary
    data["representation"] = representation_grad
    data["dynamic"] = dynamic_grad
    data["reward"] = rwd_grad
    data["policy"] = policy_grad
    data["value"] = value_grad
    logging.info(
        f"Collected gradients for all networks: \n Representation gradients: {representation_grad} \n Dynamic gradients: {dynamic_grad} \n Reward gradients: {rwd_grad} \n Policy gradients: {policy_grad} \n Value gradients: {value_grad}"
    )
    return data


def set_seed(seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    setup_logger(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timestamp", type=str, required=True, help="Timestamp of the model to analyze"
    )
    parser.add_argument(
        "--N", type=int, default=3, help="Number of disks for Towers of Hanoi"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Maximum number of steps in the environment",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--state", type=str, default="0_0_0", help="Initial state of the environment"
    )
    parser.add_argument(
        "--action", type=str, default="0", help="Action to take in the environment"
    )
    args = parser.parse_args()
    N = args.N
    max_steps = args.max_steps
    timestamp = args.timestamp
    seed = args.seed
    state = tuple(map(int, args.state.split("_")))
    action = args.action

    env = TowersOfHanoi(N=N, max_steps=max_steps)
    s_space_size = env.oneH_s_size

    set_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    command_line = (
        f"Seed: {seed}, N: {N}, Max Steps: {max_steps}, Timestamp: {timestamp}"
    )
    logging.info(command_line)

    ## Load model
    mcts = MCTS(
        n_simulations=25, root_dirichlet_alpha=0.25, discount=0.8, batch_s=1, device=dev
    )
    networks = MuZeroNet(
        action_s=6, lr=0.002, rpr_input_s=s_space_size, device=dev, TD_return=True
    )
    model_path = os.path.join("stats", "Hanoi", timestamp, "muzero_model.pt")
    model_dict = torch.load(model_path, map_location=dev)
    networks.load_state_dict(model_dict["Muzero_net"])
    networks.optimiser.load_state_dict(model_dict["Net_optim"])

    data = get_results(env, networks, mcts, state)

    save_results(data, timestamp, state)
    visualize_gradients_subgraphs(data, timestamp, state)
    logging.info("Gradient analysis completed and results saved.")
