import argparse
import copy
import logging
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib import colors as mcolors

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


def save_results(data, file_dir):

    for network, gradients in data.items():
        file_path = os.path.join(file_dir, f"{network}_gradients.pt")
        torch.save(gradients, file_path)
        logging.info(f"Saved gradients for {network} to {file_path}")


def ablate_networks(networks, policy=False, value=False):
    """Return a copy of ``networks`` with selected heads reinitialised."""
    net_copy = copy.deepcopy(networks)
    if policy:
        net_copy.policy_net.apply(net_copy.reset_param)
    if value:
        net_copy.value_net.apply(net_copy.reset_param)
    return net_copy


def register_feature_hooks(networks, feature_maps):
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


def visualize_gradients_subgraphs(
    gradients_dict,
    state_name,
    file_dir,
    feature_maps_dict=None,
    input_sensitivities_dict=None,
):
    # Set style and colorblind-friendly palette
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )
    # Use seaborn's colorblind palette
    cb_colors = sns.color_palette("colorblind")
    num_nets = len(gradients_dict)
    net_names = list(gradients_dict.keys())

    fig, axes = plt.subplots(
        nrows=num_nets,
        ncols=5,
        figsize=(32, 5 * num_nets),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 1]},
        constrained_layout=True,
    )

    if num_nets == 1:
        axes = axes[np.newaxis, :]  # To always index as [net_idx, col]

    for ax in axes.flat:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for net_idx, net_name in enumerate(net_names):
        grads = [
            g
            for g in gradients_dict[net_name]
            if g is not None and isinstance(g, torch.Tensor)
        ]

        # --------------------------------------------------------------------
        # 1. Mean Absolute Gradient per Layer (Bar Chart)
        # --------------------------------------------------------------------
        # This bar chart shows the average absolute gradient for each layer in the network.
        # It helps to diagnose issues like vanishing or exploding gradients. A very low bar
        # might indicate that a layer is not learning (vanishing gradient), while a very
        # high bar can signal instability (exploding gradient).
        #
        # For a layer l with weights W_l, the value plotted is:
        # M_l = (1 / |W_l|) * Σ_{w ∈ W_l} |∂L/∂w|
        # where L is the loss and |W_l| is the number of parameters in the layer.
        magnitudes = [g.abs().mean().item() for g in grads]
        layers = np.arange(len(magnitudes))
        bars = axes[net_idx, 0].bar(
            layers, magnitudes, color=cb_colors[0], edgecolor="k", linewidth=0.7
        )
        axes[net_idx, 0].set_title(f"{net_name.capitalize()} – Mean Abs Grad")
        if len(magnitudes) > 0:
            max_idx = np.argmax(magnitudes)
            bars[max_idx].set_color(cb_colors[6])

        axes[net_idx, 0].set_xlabel("Layer")
        axes[net_idx, 0].set_ylabel("Mean |Grad|")

        # --------------------------------------------------------------------
        # 2. Gradient Distribution (Histogram)
        # --------------------------------------------------------------------
        # This histogram displays the distribution of all gradient values across all
        # layers of the network. A healthy distribution is often centered around zero.
        # This plot can reveal if many gradients are "stuck" at zero, which might
        # indicate a problem with "dead neurons".
        #
        # This is a histogram of the set { ∂L/∂w } for all weights w in the network.
        if grads:
            all_grads = torch.cat([g.flatten() for g in grads])
            axes[net_idx, 1].hist(
                all_grads.detach().cpu().numpy(),
                bins=80,
                color=cb_colors[1],
                edgecolor="k",
                alpha=0.7,
            )
        else:
            axes[net_idx, 1].text(0.5, 0.5, "No gradients", ha="center", va="center")
        axes[net_idx, 1].set_title(f"{net_name.capitalize()} – Grad Distribution")
        axes[net_idx, 1].set_xlabel("Gradient Value")
        axes[net_idx, 1].set_ylabel("Frequency")

        # --------------------------------------------------------------------
        # 3. Gradient Norm per Layer (Line Plot)
        # --------------------------------------------------------------------
        # This line plot shows the L2 norm of the gradients for each layer. It provides
        # a measure of the total magnitude of the gradient for each layer's weights.
        # This is another useful tool for identifying layers that might be suffering
        # from exploding gradients (very high norm).
        #
        # For a layer l, the value plotted is the L2 norm (Frobenius norm for matrices):
        # N_l = ||∂L/∂W_l||₂ = sqrt(Σ_{w ∈ W_l} (∂L/∂w)²)
        norms = [g.norm().item() for g in grads]
        axes[net_idx, 2].plot(
            layers,
            norms,
            "-o",
            color=cb_colors[2],
            markersize=8,
            linewidth=2,
            markeredgecolor="k",
        )
        axes[net_idx, 2].set_title(f"{net_name.capitalize()} – Grad Norms")
        axes[net_idx, 2].set_xlabel("Layer")
        axes[net_idx, 2].set_ylabel("||Grad|| (L2)")

        # --------------------------------------------------------------------
        # 4. Feature Map (Activation Visualization)
        # --------------------------------------------------------------------
        # This plot visualizes the activations (outputs) of the neurons in a specific
        # layer, known as a feature map. It helps to understand what features the
        # network is learning to detect at different stages. For example, it can
        # show if certain neurons are consistently inactive ("dead").
        #
        # The plot shows the activation A_l of a layer l for a given input x:
        # A_l = f(W_l * x + b_l)
        # where f is the activation function.
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
                axes[net_idx, 3].plot(
                    fmap[0].numpy(), color=cb_colors[3], linewidth=1.2
                )
            axes[net_idx, 3].set_title(f"{net_name.capitalize()} – Feature Map")
        else:
            axes[net_idx, 3].text(0.5, 0.5, "No feature map", ha="center", va="center")
        axes[net_idx, 3].set_xlabel("Unit")
        axes[net_idx, 3].set_ylabel("")

        # --------------------------------------------------------------------
        # 5. Saliency Map (Input Sensitivity)
        # --------------------------------------------------------------------
        # This bar chart, often called a saliency map, shows how sensitive the network's
        # output is to each input feature. It is calculated as the absolute value of the
        # gradient of the output with respect to the input. A high bar indicates that
        # the corresponding input feature is very influential on the outcome.
        #
        # For an output O and an input feature x_i, the saliency S_i is:
        # S_i = |∂O/∂x_i|
        if input_sensitivities_dict and net_name in input_sensitivities_dict:
            saliency = np.abs(input_sensitivities_dict[net_name])
            bars = axes[net_idx, 4].bar(
                range(len(saliency)),
                saliency,
                color=cb_colors[4],
                edgecolor="k",
                linewidth=0.6,
            )
            if len(saliency) > 0:
                max_idx = np.argmax(saliency)
                bars[max_idx].set_color(cb_colors[5])
            axes[net_idx, 4].set_title(f"{net_name.capitalize()} – Input Sensitivity")
        else:
            axes[net_idx, 4].text(0.5, 0.5, "No saliency", ha="center", va="center")
        axes[net_idx, 4].set_xlabel("Input Feature")
        axes[net_idx, 4].set_ylabel("|∂output/∂input|")

    action_str = get_action_str(action)
    fig.suptitle(
        f"Gradient Diagnostics for State: {state_name}, Action: {action_str}",
        fontsize=28,
        y=1.03,
    )
    plt.savefig(
        os.path.join(file_dir, f"gradients_subgraphs.png"),
        dpi=300,
        bbox_inches="tight",
        format="png",
    )
    plt.close(fig)


def get_results(networks, state, action):
    """
    Run the model with a given state and collect results.
    """
    data = {}
    # Set an initial state
    oneH_c_state = torch.tensor(
        oneHot_encoding(state, n_integers=env.n_pegs), dtype=torch.float32
    ).unsqueeze(0)

    # Representation
    h_state = networks.represent(oneH_c_state)

    # Prediction
    pi_logits, value = networks.prediction(h_state)

    pi_probs = F.softmax(pi_logits, dim=-1)

    dummy_action = torch.zeros(1, networks.num_actions, device=networks.dev)
    dummy_action[0, action] = 1.0

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


def get_action_str(action_int):
    """Converts an action integer to a human-readable string."""
    action_map = {
        0: "A -> B",
        1: "A -> C",
        2: "B -> A",
        3: "B -> C",
        4: "C -> A",
        5: "C -> B",
    }
    return action_map.get(action_int, f"Action {action_int}")


def compute_saliency(state, N, networks, action):
    """
    Compute per-head saliency by re-running the minimal forward+backward
    for each network component.
    """
    # Precompute numpy state and dummy action
    state_np = oneHot_encoding(state, n_integers=env.n_pegs)
    dummy_action = torch.zeros(1, networks.num_actions, device=networks.dev)
    dummy_action[0, action] = 1.0

    # Define each head as a function mapping input→scalar
    heads = {
        "representation": lambda x: networks.represent(x).mean(),
        "dynamic": lambda x: networks.dynamics(networks.represent(x), dummy_action)[
            0
        ].mean(),
        "reward": lambda x: networks.dynamics(networks.represent(x), dummy_action)[
            1
        ].mean(),
        "policy": lambda x: (lambda logits: logits[0, logits.argmax(dim=1).item()])(
            networks.prediction(networks.represent(x))[0]
        ),
        "value": lambda x: networks.prediction(networks.represent(x))[1].mean(),
    }

    saliencies = {}
    for name, fn in heads.items():
        # fresh input tensor
        inp = torch.tensor(state_np, dtype=torch.float32, device=networks.dev)
        inp = inp.unsqueeze(0).requires_grad_()

        networks.zero_grad()
        out = fn(inp)  # forward up to this head
        out.backward()  # compute ∂out/∂inp

        saliencies[name] = inp.grad.detach().cpu().numpy()[0]

    return saliencies


def visualize_hanoi_state(ax, state, title, saliency_per_disk=None):
    """
    Visualize the Hanoi state, optionally adjusting disk appearance for saliency.
    """
    rod_positions = [0, 1, 2]
    disk_visual_props = {
        "large": {"width": 0.8, "color": "firebrick", "label": "Large"},
        "medium": {"width": 0.6, "color": "royalblue", "label": "Medium"},
        "small": {"width": 0.4, "color": "forestgreen", "label": "Small"},
    }
    disk_order = ["large", "medium", "small"]

    normalized_saliencies = {}
    if saliency_per_disk:
        saliency_dict = {
            "small": saliency_per_disk[0],
            "medium": saliency_per_disk[1],
            "large": saliency_per_disk[2],
        }
        min_saliency, max_saliency = min(saliency_dict.values()), max(
            saliency_dict.values()
        )
        if max_saliency == min_saliency:
            normalized_saliencies = {disk: 0.5 for disk in disk_order}
        else:
            for disk in disk_order:
                score = saliency_dict[disk]
                normalized_saliencies[disk] = (score - min_saliency) / (
                    max_saliency - min_saliency
                )

    ax.clear()
    ax.hlines(0, -0.5, 2.5, colors="black", linewidth=3)
    for rod_pos in rod_positions:
        ax.vlines(rod_pos, 0, 3.5, colors="black", linewidth=2)

    rod_counts = {0: 0, 1: 0, 2: 0}

    for disk_name in disk_order:
        disk_map = {"small": 0, "medium": 1, "large": 2}
        rod = state[disk_map[disk_name]]

        y_pos = 0.3 + rod_counts[rod] * 0.4
        props = disk_visual_props[disk_name]

        norm_saliency = 0.0
        alpha_val, linewidth_val, edge_color = 1.0, 1.5, "black"
        face_color = props["color"]
        if saliency_per_disk:
            norm_saliency = normalized_saliencies[disk_name]
            intensity = 0.5 + norm_saliency * 0.5
            base_col = np.array(mcolors.to_rgb(props["color"]))
            face_colour = tuple(intensity * base_col + (1 - intensity) * np.ones(3))
            linewidth_val = 1.5 + norm_saliency * 3.5
            if norm_saliency > 0.95:
                edge_color = "gold"

        disk_text = f"{props['label'][0]}: {norm_saliency:.2f}"

        rect = plt.Rectangle(
            (rod - props["width"] / 2, y_pos),
            props["width"],
            0.3,
            facecolor=face_colour,
            edgecolor=edge_color,
            linewidth=linewidth_val,
            alpha=alpha_val,
        )
        ax.add_patch(rect)
        ax.text(
            rod,
            y_pos + 0.15,
            disk_text,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

        rod_counts[rod] += 1

    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.1, 4.0)
    ax.set_xticks(rod_positions)
    ax.set_xticklabels(["A", "B", "C"])
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)


def aggregate_saliency_per_disk(saliency_vector, num_disks=3, num_rods=3):
    """
    Aggregates a flat saliency vector into a per-disk saliency list.

    Args:
        saliency_vector (np.array): The flat saliency map from the model.
        num_disks (int): The number of disks in the puzzle.
        num_rods (int): The number of rods in the puzzle.

    Returns:
        list: A list of saliency scores, ordered [small, medium, large].
    """
    if len(saliency_vector) != num_disks * num_rods:
        raise ValueError("Saliency vector length does not match num_disks * num_rods.")

    saliency_per_disk = []
    # Assumes the input vector is ordered by disk size (small, medium, large)
    for i in range(num_disks):
        start_index = i * num_rods
        end_index = start_index + num_rods
        disk_saliency = np.sum(np.abs(saliency_vector[start_index:end_index]))
        saliency_per_disk.append(disk_saliency)

    return saliency_per_disk


def visualize_saliency_comparison(saliency_data, state, file_dir):
    """
    Creates and saves a subplot comparing saliency diagrams for each MuZero network.

    Args:
        saliency_data (dict): A dictionary where keys are network names (e.g., 'policy', 'value')
                              and values are the raw saliency vectors from the model.
        state (tuple): The game state for which the saliency was calculated.
        save_path (str): The file path to save the resulting image (e.g., 'comparison.png').
    """
    net_names = list(saliency_data.keys())
    num_nets = len(net_names)
    action_str = get_action_str(action)

    # Create a subplot grid: 1 row, columns equal to the number of networks
    fig, axes = plt.subplots(
        1,
        num_nets,
        figsize=(5 * num_nets, 5.5),  # Adjust size for readability
        constrained_layout=True,
    )

    # If there's only one network, axes is not a list, so we make it one
    if num_nets == 1:
        axes = [axes]

    # Iterate through each network's saliency data
    for i, net_name in enumerate(net_names):
        raw_saliency_vector = saliency_data[net_name]
        ax = axes[i]

        # 1. Aggregate the raw saliency vector to get one score per disk
        saliency_scores = aggregate_saliency_per_disk(raw_saliency_vector)

        # 2. Use the visualization function to draw the hanoi state on the subplot axis
        visualize_hanoi_state(
            ax=ax,
            state=state,
            title=f"{net_name.capitalize()} Saliency",
            saliency_per_disk=saliency_scores,
        )

    # Add a main title to the entire figure
    fig.suptitle(
        f"Saliency Map for State: {state}, Action: {action_str}", fontsize=20, y=1.05
    )

    save_path = os.path.join(file_dir, "gradients_hanoi_diagram.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Comparison subplot saved to: %s", save_path)


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
    parser.add_argument(
        "--compare_ablations",
        type=bool,
        default=True,
        help="Also compute gradients for policy and value ablated networks",
    )
    args = parser.parse_args()
    N = args.N
    max_steps = args.max_steps
    timestamp = args.timestamp
    seed = args.seed
    state = tuple(map(int, args.state.split("_")))
    action = int(args.action)

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
    register_feature_hooks(networks, feature_maps)
    results = []

    data = get_results(networks, state, action)
    saliency = compute_saliency(state, N, networks, action)
    results.append(("baseline", feature_maps, data, saliency))

    if args.compare_ablations:
        ablation_settings = {
            "policy_ablated": {"policy": True},
            "value_ablated": {"value": True},
        }
        for name, opts in ablation_settings.items():
            ab_net = ablate_networks(networks, **opts)
            fmap_ab = {}
            register_feature_hooks(ab_net, fmap_ab)
            data_ab = get_results(ab_net, state, action)
            saliency_ab = compute_saliency(state, N, ab_net, action)
            results.append((name, fmap_ab, data_ab, saliency_ab))

    base_dir = os.path.join(
        "stats", "Hanoi", timestamp, "gradients", f"{args.state}-{args.action}"
    )
    for label, fmap, data_dict, sal in results:
        file_dir = os.path.join(base_dir, label)
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Collected features for {label}: {fmap}")
        save_results(data_dict, file_dir)
        visualize_gradients_subgraphs(data_dict, args.state, file_dir, fmap, sal)
        visualize_saliency_comparison(saliency_data=sal, state=state, file_dir=file_dir)

    logging.info("Gradient analysis completed and results saved.")
