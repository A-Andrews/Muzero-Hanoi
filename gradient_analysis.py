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


def get_activation(name, fmap_dict):
    def hook(model, input, output):
        fmap_dict[name] = output.detach().cpu()

    return hook


def save_results(data, timestamp, state):
    state_name = "_".join(map(str, state))
    file_dir = os.path.join("stats", "Hanoi", timestamp, "gradients", state_name)
    os.makedirs(file_dir, exist_ok=True)

    for network, gradients in data.items():
        file_path = os.path.join(file_dir, f"{network}_gradients.pt")
        torch.save(gradients, file_path)
        logging.info(f"Saved gradients for {network} to {file_path}")


import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns


def visualize_gradients_subgraphs(
    gradients_dict, timestamp, state, feature_maps_dict=None, input_sensitivities_dict=None
):
    # Set style and colorblind-friendly palette
    plt.rcParams.update({
        "font.family": "serif",
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })
    # Use seaborn's colorblind palette
    cb_colors = sns.color_palette("colorblind")
    state_name = "_".join(map(str, state))
    file_dir = os.path.join("stats", "Hanoi", timestamp, "gradients", state_name)
    os.makedirs(file_dir, exist_ok=True)
    num_nets = len(gradients_dict)
    net_names = list(gradients_dict.keys())

    fig, axes = plt.subplots(
        nrows=num_nets,
        ncols=5,
        figsize=(32, 5 * num_nets),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 1]},
        constrained_layout=True
    )

    if num_nets == 1:
        axes = axes[np.newaxis, :]  # To always index as [net_idx, col]

    for ax in axes.flat:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for net_idx, net_name in enumerate(net_names):
        grads = [
            g
            for g in gradients_dict[net_name]
            if g is not None and isinstance(g, torch.Tensor)
        ]
        # 1. Mean Absolute Gradient per Layer (Bar)
        magnitudes = [g.abs().mean().item() for g in grads]
        layers = np.arange(len(magnitudes))
        bars = axes[net_idx, 0].bar(
            layers, magnitudes, color=cb_colors[0], edgecolor="k", linewidth=0.7
        )
        axes[net_idx, 0].set_title(f"{net_name.capitalize()} – Mean Abs Grad")
        if len(magnitudes) > 0:
                max_idx = np.argmax(magnitudes)
                bars[max_idx].set_color(cb_colors[6])

        if net_idx == num_nets - 1:
            axes[net_idx, 0].set_xlabel("Layer")
        else:
            axes[net_idx, 0].set_xlabel("")
            axes[net_idx, 0].set_xticklabels([])
        axes[net_idx, 0].set_ylabel("Mean |Grad|")

        # 2. Gradient Distribution (Histogram)
        if grads:
            all_grads = torch.cat([g.flatten() for g in grads])
            axes[net_idx, 1].hist(
                all_grads.detach().cpu().numpy(), bins=80, color=cb_colors[1], edgecolor="k", alpha=0.7
            )
        else:
            axes[net_idx, 1].text(0.5, 0.5, "No gradients", ha="center", va="center")
        axes[net_idx, 1].set_title(f"{net_name.capitalize()} – Grad Distribution")
        if net_idx == num_nets - 1:
            axes[net_idx, 1].set_xlabel("Gradient Value")
        else:
            axes[net_idx, 1].set_xlabel("")
            axes[net_idx, 1].set_xticklabels([])
        axes[net_idx, 1].set_ylabel("Frequency")

        # 3. Gradient Norm per Layer (Line)
        norms = [g.norm().item() for g in grads]
        axes[net_idx, 2].plot(
            layers, norms, "-o", color=cb_colors[2], markersize=8, linewidth=2, markeredgecolor="k"
        )
        axes[net_idx, 2].set_title(f"{net_name.capitalize()} – Grad Norms")
        if net_idx == num_nets - 1:
            axes[net_idx, 2].set_xlabel("Layer")
        else:
            axes[net_idx, 2].set_xlabel("")
            axes[net_idx, 2].set_xticklabels([])
        axes[net_idx, 2].set_ylabel("||Grad|| (L2)")

        # 4. Feature Map
        layer_names = [f"{net_name}_linear1", f"{net_name}_linear2"]
        fmap = None
        for name in layer_names:
            if feature_maps_dict and name in feature_maps_dict:
                fmap = feature_maps_dict[name]
                break
        if fmap is not None:
            if fmap.ndim == 4:
                axes[net_idx, 3].imshow(
                    fmap[0, 0].numpy(), cmap="cividis", aspect="auto"
                )
            elif fmap.ndim == 2:
                axes[net_idx, 3].plot(fmap[0].numpy(), color=cb_colors[3], linewidth=1.2)
            axes[net_idx, 3].set_title(f"{net_name.capitalize()} – Feature Map")
        else:
            axes[net_idx, 3].text(0.5, 0.5, "No feature map", ha="center", va="center")
        if net_idx == num_nets - 1:
            axes[net_idx, 3].set_xlabel("Unit")
        else:
            axes[net_idx, 3].set_xlabel("")
            axes[net_idx, 3].set_xticklabels([])
        axes[net_idx, 3].set_ylabel("")

        # 5. Saliency Map (Input Sensitivity)
        if input_sensitivities_dict and net_name in input_sensitivities_dict:
            saliency = np.abs(input_sensitivities_dict[net_name])
            bars = axes[net_idx, 4].bar(
                range(len(saliency)), saliency, color=cb_colors[4], edgecolor="k", linewidth=0.6
            )
            if len(saliency) > 0:
                max_idx = np.argmax(saliency)
                bars[max_idx].set_color(cb_colors[5])
            axes[net_idx, 4].set_title(f"{net_name.capitalize()} – Input Sensitivity")
        else:
            axes[net_idx, 4].text(0.5, 0.5, "No saliency", ha="center", va="center")
        if net_idx == num_nets - 1:
            axes[net_idx, 4].set_xlabel("Input Feature")
        else:
            axes[net_idx, 4].set_xlabel("")
            axes[net_idx, 4].set_xticklabels([])
        axes[net_idx, 4].set_ylabel("|∂output/∂input|")

    fig.suptitle(
        f"MuZero Gradient & Feature Diagnostics for State: {state_name}", fontsize=28, y=1.03
    )
    plt.savefig(
        os.path.join(file_dir, f"gradients_subgraphs.png"),
        dpi=300,
        bbox_inches="tight",
        format="png",
    )
    plt.close(fig)




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

def compute_saliency(state, N, networks):
    input_sensitivities = {}

    # Make sure your input is a leaf and requires grad
    oneH_c_state_np = oneHot_encoding(state, n_integers=N)
    oneH_c_state = torch.tensor(oneH_c_state_np, dtype=torch.float32, device=networks.dev)
    oneH_c_state = oneH_c_state.unsqueeze(0)
    oneH_c_state.requires_grad_()

    # Forward pass
    h_state = networks.represent(oneH_c_state)
    pi_logits, value = networks.prediction(h_state)
    rwd = torch.zeros_like(value)
    dummy_action = torch.zeros(1, networks.num_actions, device=networks.dev)
    dummy_action[0, 0] = 1.0
    next_h_state, next_rwd = networks.dynamics(h_state, dummy_action)

    # For each head, compute saliency (input gradient)
    outputs = {
        "representation": h_state.mean(),
        "dynamic": next_h_state.mean(),
        "reward": next_rwd.mean(),
        "policy": pi_logits[0, pi_logits.argmax(dim=1).item()],  # Best action logit
        "value": value.mean(),
    }

    for net_name, scalar_output in outputs.items():
        networks.zero_grad()
        if oneH_c_state.grad is not None:
            oneH_c_state.grad.zero_()
        scalar_output.backward(retain_graph=True)
        # Save abs gradient (saliency) for this head
        input_sensitivities[net_name] = oneH_c_state.grad.detach().cpu().numpy()[0]

    return input_sensitivities



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

    feature_maps = {}
    networks.representation_net[0].register_forward_hook(
        get_activation("representation_linear1", feature_maps)
    )
    networks.representation_net[2].register_forward_hook(
        get_activation("representation_linear2", feature_maps)
    )
    networks.dynamic_net[0].register_forward_hook(
        get_activation("dynamic_linear1", feature_maps)
    )
    networks.dynamic_net[2].register_forward_hook(
        get_activation("dynamic_linear2", feature_maps)
    )
    networks.rwd_net[0].register_forward_hook(
        get_activation("reward_linear1", feature_maps)
    )
    networks.rwd_net[2].register_forward_hook(
        get_activation("reward_linear2", feature_maps)
    )
    networks.policy_net[0].register_forward_hook(
        get_activation("policy_linear1", feature_maps)
    )
    networks.policy_net[2].register_forward_hook(
        get_activation("policy_linear2", feature_maps)
    )
    networks.value_net[0].register_forward_hook(
        get_activation("value_linear1", feature_maps)
    )
    networks.value_net[2].register_forward_hook(
        get_activation("value_linear2", feature_maps)
    )

    data = get_results(env, networks, mcts, state)
    saliency = compute_saliency(state, N, networks)

    logging.info(f"Collected features: {feature_maps}")
    save_results(data, timestamp, state)
    visualize_gradients_subgraphs(data, timestamp, state, feature_maps, saliency)
    logging.info("Gradient analysis completed and results saved.")
