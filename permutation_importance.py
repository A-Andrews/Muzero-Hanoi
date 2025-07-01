import argparse
import copy
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from env.hanoi import TowersOfHanoi
from MCTS.mcts import MCTS
from networks import MuZeroNet
from utils import PLOT_COLORS, oneHot_encoding, set_plot_style, setup_logger


def get_ablated_network(networks, ablate_policy=False, ablate_value=False):
    """Return a copy of ``networks`` with selected components reinitialised."""
    ablated = copy.deepcopy(networks)
    if ablate_policy:
        ablated.policy_net.apply(ablated.reset_param)
    if ablate_value:
        ablated.value_net.apply(ablated.reset_param)
    return ablated


def play_game(networks, env, start_state, max_game_steps, mcts, permute_feature=None):
    """
    Plays a single game of Towers of Hanoi from a start state.

    Args:
        networks (MuZeroNet): The trained neural network.
        env (TowersOfHanoi): The game environment.
        start_state (tuple): The starting configuration of the disks.
        max_game_steps (int): The maximum number of moves before giving up.
        mcts (MCTS): The Monte Carlo Tree Search algorithm instance.
        permute_feature (int, optional): The index of the feature (disk) to permute.
                                         0 for small, 1 for medium, 2 for large.
                                         If None, normal gameplay occurs.

    Returns:
        bool: True if the puzzle was solved, False otherwise.
    """
    env.reset()

    env.c_state = start_state
    env.oneH_c_state = oneHot_encoding(env.c_state, n_integers=env.n_pegs)

    for _ in range(max_game_steps):
        true_state = env.c_state
        state_to_feed_model = true_state

        # --- PERMUTATION INJECTION POINT ---
        # If a feature is to be permuted, corrupt the state before feeding it to the model.
        if permute_feature is not None:
            permuted_state_list = list(true_state)
            # Replace the true position of the specified disk with a random one.
            permuted_state_list[permute_feature] = np.random.randint(0, env.n_pegs)
            state_to_feed_model = tuple(permuted_state_list)

        state_one_hot = oneHot_encoding(state_to_feed_model, n_integers=env.n_pegs)

        # Let the agent decide on a move based on the (potentially corrupted) state
        action, _, _ = mcts.run_mcts(
            state_one_hot, networks, temperature=0, deterministic=True
        )

        # Apply the chosen action to the REAL environment
        _, reward, done, _ = env.step(action)

        if reward == 100:
            return True
        if done:
            return False

    return False


def run_evaluation(
    networks, env, test_states, max_game_steps, mcts, permute_feature=None
):
    """
    Runs the agent over a set of test states and calculates the solve rate.
    """
    solved_count = 0
    # Use tqdm for a nice progress bar
    for start_state in tqdm(test_states, desc=f"Permuting '{permute_feature}'"):
        if play_game(networks, env, start_state, max_game_steps, mcts, permute_feature):
            solved_count += 1

    return solved_count / len(test_states)


def main():
    parser = argparse.ArgumentParser(
        description="Run Permutation Importance analysis on a MuZero model for Towers of Hanoi."
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        required=True,
        help="Timestamp of the model to analyze.",
    )
    parser.add_argument(
        "--N", type=int, default=3, help="Number of disks for Towers of Hanoi."
    )
    parser.add_argument(
        "--max_game_steps", type=int, default=50, help="Max moves per game."
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    # --- Setup ---
    setup_logger(args.seed)
    set_plot_style()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TowersOfHanoi(N=args.N, max_steps=args.max_game_steps)

    # Load the trained model
    logging.info(f"Loading model with timestamp: {args.timestamp}")
    mcts = MCTS(
        n_simulations=25, root_dirichlet_alpha=0.25, discount=0.8, batch_s=1, device=dev
    )
    networks = MuZeroNet(
        action_s=6,
        lr=0.002,
        rpr_input_s=env.oneH_s_size,
        device=dev,
        TD_return=True,
    )
    model_path = os.path.join("stats", "Hanoi", args.timestamp, "muzero_model.pt")
    model_dict = torch.load(model_path, map_location=dev)
    networks.load_state_dict(model_dict["Muzero_net"])
    networks.eval()  # Set model to evaluation mode

    # --- Define the Evaluation Set ---
    # A fixed set of diverse starting states for consistent evaluation
    test_states = [
        (0, 0, 0),
        (1, 1, 1),
        (0, 1, 2),
        (2, 1, 0),
        (0, 0, 1),
        (1, 0, 0),
        (2, 2, 0),
        (1, 2, 1),
        (0, 2, 2),
        (2, 0, 1),
        (1, 1, 0),
        (0, 2, 1),
    ]
    logging.info(f"Using a test set of {len(test_states)} starting states.")

    disk_names = ["Small Disk", "Medium Disk", "Large Disk"]

    experiment_networks = {
        "baseline": networks,
        "policy_ablated": get_ablated_network(networks, ablate_policy=True),
        "value_ablated": get_ablated_network(networks, ablate_value=True),
    }

    results = {}
    for label, net in experiment_networks.items():
        logging.info(f"Evaluating {label} network...")
        base_rate = run_evaluation(
            net, env, test_states, args.max_game_steps, mcts, permute_feature=None
        )
        logging.info(f"  {label} baseline solve rate: {base_rate:.2%}")

        importances = {}
        for i, disk_name in enumerate(disk_names):
            logging.info(f"  Permuting {disk_name} for {label}...")
            permuted_rate = run_evaluation(
                net, env, test_states, args.max_game_steps, mcts, permute_feature=i
            )
            logging.info(f"    -> Solve Rate: {permuted_rate:.2%}")
            importances[disk_name] = base_rate - permuted_rate
        results[label] = importances

    # --- Report and Visualize Results ---
    logging.info("\n--- Permutation Importance Results ---")
    for label, importances in results.items():
        logging.info(f"Results for {label}:")
        for disk, imp in importances.items():
            logging.info(f"  {disk}: {imp:.4f}")

    # Create and save a grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(disk_names))
    width = 0.25
    labels = list(results.keys())
    for idx, label in enumerate(labels):
        offsets = x + (idx - (len(labels) - 1) / 2) * width
        values = [results[label][d] for d in disk_names]
        bars = ax.bar(
            offsets,
            values,
            width,
            label=label.replace("_", " "),
            color=PLOT_COLORS[idx % len(PLOT_COLORS)],
            edgecolor="none",
        )
        ax.bar_label(bars, fmt="%.3f", padding=3)

    ax.set_ylabel("Importance (Drop in Solve Rate)")
    ax.set_title("Permutation Importance of Disk Positions for Hanoi Agent")
    ax.set_xticks(x)
    ax.set_xticklabels(disk_names)
    ax.set_ylim(bottom=0)
    ax.legend()
    plt.tight_layout()

    save_dir = os.path.join("stats", "Hanoi", args.timestamp)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "permutation_importance.png")
    plt.savefig(save_path)
    logging.info(f"Importance plot saved to {save_path}")


if __name__ == "__main__":
    main()
