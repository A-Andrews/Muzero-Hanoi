import argparse
import logging
import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.profiler
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

from env.hanoi import TowersOfHanoi
from Muzero import Muzero
from utils import setup_logger


def get_env(env_name):
    if env_name == "Hanoi":
        N = 3
        max_steps = 200
        env = TowersOfHanoi(N=N, max_steps=max_steps)
        s_space_size = env.oneH_s_size
        n_action = 6  # n. of action available in each state for Tower of Hanoi (including illegal ones)
    else:  # Use for gym env with discrete 1d action space
        env = gym.make(env_name)
        assert isinstance(
            env.action_space, gym.spaces.discrete.Discrete
        ), "Must be discrete action space"
        s_space_size = env.observation_space.shape[0]
        n_action = env.action_space.n
        max_steps = env.spec.max_episode_steps
        N = None
    return env, s_space_size, n_action, max_steps, N


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    setup_logger(seed)


def log_command_line(
    env_p,
    training_loops,
    min_replay_size,
    lr,
    discount,
    n_mcts_simulations,
    batch_s,
    TD_return,
    priority_replay,
    dev,
    n_disks=None,
):
    command_line = f"Env: {env_p}, Training Loops: {training_loops}, Min replay size: {min_replay_size}, lr: {lr}, discount: {discount}, n. MCTS: {n_mcts_simulations}, batch size: {batch_s}, TD_return: {TD_return}, Priority Buff: {priority_replay}, device: {dev}"
    if env_p == "Hanoi":  # if hanoi also print n. of disks
        command_line += f", N. disks: {n_disks}"
    logging.info(command_line)
    return command_line


## ======== Initialise alg. ========


# scheduler = ReduceLROnPlateau(muzero.networks.optimiser, patience=50, factor=0.5)


def analyze_state_decisions(stats):
    """Analyze the average decisions per state"""
    avg_decisions = stats["avg_decisions_per_state"]

    logging.info(f"Number of unique states encountered: {len(avg_decisions)}")
    logging.info(
        f"Average action across all states: {np.mean(list(avg_decisions.values())):.2f}"
    )
    logging.info(
        f"Action variance across states: {np.var(list(avg_decisions.values())):.2f}"
    )

    # Find states with highest and lowest average actions
    max_state = max(avg_decisions, key=avg_decisions.get)
    min_state = min(avg_decisions, key=avg_decisions.get)

    logging.info(f"State with highest avg action: {avg_decisions[max_state]:.2f}")
    logging.info(f"State with lowest avg action: {avg_decisions[min_state]:.2f}")
    return avg_decisions


def plot_action_distribution(
    stats, env_name, buffer_size, n_mcts_simulations, lr, unroll_n_steps, timestamp
):
    """Plot distribution of average actions per state"""
    avg_decisions = list(stats["avg_decisions_per_state"].values())

    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_dir = os.path.join(file_dir, "img", "training_performance", env_name)
    os.makedirs(file_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(avg_decisions, bins=20, alpha=0.7)
    plt.xlabel("Average Action")
    plt.ylabel("Number of States")
    plt.title("Distribution of Average Actions per State")
    plt.grid(True)
    plot_path = os.path.join(
        file_dir,
        f"action_distribution_buff-{buffer_size}_mctss-{n_mcts_simulations}_lr-{lr}_unroll-{unroll_n_steps}_{timestamp}.png",
    )
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Action distribution plot saved to {plot_path}")


## ===== Save results =========
def save_results(env_name, command_line, stats, muzero, timestamp):
    file_indx = 1
    # Create directory to store results
    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_dir = os.path.join(file_dir, "stats", env_name, str(timestamp))

    # Create directory if it did't exist before
    os.makedirs(file_dir, exist_ok=True)

    # Store command line
    with open(os.path.join(file_dir, "commands.txt"), "w") as f:
        f.write(command_line)

    dict_keys = ["state_action_history", "avg_decisions_per_state"]

    # Save all stats
    for key, value in stats.items():
        if key in dict_keys:
            # Save dictionaries directly without converting to tensor
            torch.save(value, os.path.join(file_dir, f"{key}-{timestamp}.pt"))
        else:
            # Convert other values to tensors
            torch.save(
                torch.tensor(value), os.path.join(file_dir, f"{key}-{timestamp}.pt")
            )

    model_dir = os.path.join(file_dir, "muzero_model.pt")
    # Store model
    torch.save(
        {
            "Muzero_net": (
                muzero.networks._orig_mod.state_dict()
                if hasattr(muzero.networks, "_orig_mod")
                else muzero.networks.state_dict()
            ),
            "Net_optim": muzero.networks.optimiser.state_dict(),
        },
        model_dir,
    )
    logging.info(f"Model saved to {model_dir}")


## ===== Plot results ========
def plot_results(
    stats, env_name, buffer_size, n_mcts_simulations, lr, unroll_n_steps, timestamp
):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_dir = os.path.join(file_dir, "stats", env_name, str(timestamp))
    os.makedirs(file_dir, exist_ok=True)

    for k, v in stats.items():
        if torch.is_tensor(v):
            stats[k] = v.detach().cpu().numpy()
        elif isinstance(v, list) and len(v) > 0:
            stats[k] = [
                x.detach().cpu().numpy() if torch.is_tensor(x) else x for x in v
            ]

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.plot(stats["tot_accuracy"], marker="o")
    plt.title("Mean Accuracy")
    plt.xlabel("Training Loop")
    plt.ylabel("Accuracy")

    plt.subplot(2, 3, 2)
    plt.plot(stats["all_value_loss"], label="Value Loss")
    plt.title("Value Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")

    plt.subplot(2, 3, 3)
    plt.plot(stats["all_rwd_loss"], label="Reward Loss", color="orange")
    plt.title("Reward Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")

    plt.subplot(2, 3, 4)
    plt.plot(stats["all_pi_loss"], label="Policy Loss", color="green")
    plt.title("Policy Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")

    plt.subplot(2, 3, 5)
    plt.plot(stats["cumulative_success"], label="Cumulative Success", color="purple")
    plt.title("Cumulative Success")
    plt.xlabel("Training Loop")
    plt.ylabel("Total Successes")

    # plt.subplot(2, 4, 8)
    # plt.plot(stats["q_values_per_loop"], label="Mean Q Value", color='magenta')
    # plt.title("Mean Q Value per Loop")
    # plt.xlabel("Training Loop")
    # plt.ylabel("Q Value")

    plt.subplot(2, 3, 6)
    plt.plot(stats["grad_norms"], label="Gradient Norm", color="brown")
    plt.title("Gradient Norm Over Time")
    plt.xlabel("Update Step")
    plt.ylabel("L2 Norm")

    plt.tight_layout()
    plt.suptitle(
        f"{env_name} Buffer: {buffer_size} Number of simulations: {n_mcts_simulations} Learning rate: {lr} Unroll steps: {unroll_n_steps}",
        fontsize=30,
    )
    plot_path = os.path.join(
        file_dir,
        f"training_stats_buff-{buffer_size}_mctss-{n_mcts_simulations}_lr-{lr}_unroll-{unroll_n_steps}_{timestamp}.png",
    )
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Training stats plot saved to {plot_path}")
    # TODO save all variables in plot postscript?


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Muzero Training")
    parser.add_argument(
        "--env",
        type=str,
        default="Hanoi",
        help="Environment to train on (h for Hanoi, c for CartPole)",
    )
    parser.add_argument(
        "--training_loops", type=int, default=5000, help="Number of training loops"
    )
    parser.add_argument(
        "--min_replay_size", type=int, default=5000, help="Minimum replay size"
    )
    parser.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=0.25,
        help="Dirichlet alpha for exploration",
    )
    parser.add_argument(
        "--n_ep_x_loop", type=int, default=1, help="Number of episodes per loop"
    )
    parser.add_argument(
        "--n_update_x_loop", type=int, default=1, help="Number of updates per loop"
    )
    parser.add_argument(
        "--unroll_n_steps", type=int, default=5, help="Number of unroll steps"
    )
    parser.add_argument("--TD_return", type=bool, default=True, help="Use TD return")
    parser.add_argument("--n_TD_step", type=int, default=10, help="Number of TD steps")
    parser.add_argument("--buffer_size", type=int, default=50000, help="Buffer size")
    parser.add_argument(
        "--priority_replay", type=bool, default=True, help="Use priority replay"
    )
    parser.add_argument("--batch_s", type=int, default=256, help="Batch size")
    parser.add_argument("--discount", type=float, default=0.8, help="Discount factor")
    parser.add_argument(
        "--n_mcts_simulations", type=int, default=25, help="Number of MCTS simulations"
    )
    parser.add_argument("--lr", type=float, default=0.02, help="Learning rate")
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for reproducibility"
    )
    parser.add_argument("--profile", type=bool, default=False, help="Enable profiling")
    args = parser.parse_args()
    env_p = args.env
    training_loops = args.training_loops
    min_replay_size = args.min_replay_size
    dirichlet_alpha = args.dirichlet_alpha
    n_ep_x_loop = args.n_ep_x_loop
    n_update_x_loop = args.n_update_x_loop
    unroll_n_steps = args.unroll_n_steps
    TD_return = args.TD_return
    n_TD_step = args.n_TD_step
    buffer_size = args.buffer_size
    priority_replay = args.priority_replay
    batch_s = args.batch_s
    discount = args.discount
    n_mcts_simulations = args.n_mcts_simulations
    lr = args.lr
    s = args.seed

    # Set the seed
    set_seed(s)

    # Set the device
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dev = torch.device("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")

    ## ========= Initialise env ========
    env, s_space_size, n_action, max_steps, n_disks = get_env(env_p)

    command_line = log_command_line(
        env_p,
        training_loops,
        min_replay_size,
        lr,
        discount,
        n_mcts_simulations,
        batch_s,
        TD_return,
        priority_replay,
        dev,
        n_disks,
    )

    ## ======== Initialise alg. ========
    muzero = Muzero(
        env=env,
        s_space_size=s_space_size,
        n_action=n_action,
        discount=discount,
        dirichlet_alpha=dirichlet_alpha,
        n_mcts_simulations=n_mcts_simulations,
        unroll_n_steps=unroll_n_steps,
        batch_s=batch_s,
        TD_return=TD_return,
        n_TD_step=n_TD_step,
        lr=lr,
        buffer_size=buffer_size,
        priority_replay=priority_replay,
        device=dev,
        n_ep_x_loop=n_ep_x_loop,
        n_update_x_loop=n_update_x_loop,
    )

    stats = muzero.training_loop(training_loops, min_replay_size)
    timestamp = int(time.time())
    save_results(env_p, command_line, stats, muzero, timestamp)
    plot_results(
        stats, env_p, buffer_size, n_mcts_simulations, lr, unroll_n_steps, timestamp
    )

    analyze_state_decisions(stats)
    plot_action_distribution(
        stats, env_p, buffer_size, n_mcts_simulations, lr, unroll_n_steps, timestamp
    )
