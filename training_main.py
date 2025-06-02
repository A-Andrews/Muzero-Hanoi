import argparse
import logging
import os
import time

import gym
import numpy as np
import torch

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


## ======= Set seeds for debugging =======
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    setup_logger(seed)


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


if __name__ == "__main__":
    
    set_seed(1)

    ## ======= Select the environment ========
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for reproducibility"
    )
    parser.add_argument("--profile", type=bool, default=False, help="Enable profiling")
    args = parser.parse_args()
    env_name = args.env
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

    ## ========= Useful variables: ===========

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ## ========= Initialise env ========
    env, s_space_size, n_action, max_steps, n_disks = get_env(env_name)

    ## ====== Log command line =====

    command_line = log_command_line(
        env_name,
        training_loops,
        min_replay_size,
        lr,
        discount,
        n_mcts_simulations,
        batch_s,
        TD_return,
        priority_replay,
        dev,
        n_disks=n_disks,
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

    ## ======== Run training ==========
    tot_acc = muzero.training_loop(training_loops, min_replay_size)

    save_results(
        env_name, command_line, {"total_accuracy": tot_acc}, muzero, int(time.time())
    )
