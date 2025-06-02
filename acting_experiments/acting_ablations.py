import argparse
import sys

sys.path.append("/well/costa/users/zqa082/Muzero-Hanoi")
import logging
import os

import numpy as np
import torch

from env.hanoi import TowersOfHanoi
from env.hanoi_utils import hanoi_solver
from MCTS.mcts import MCTS
from networks import MuZeroNet
from utils import setup_logger

""" Ablate different components of a trained Muzero agent to assess their impact on planning across different task difficulties (ES, MS, LS) """

## ======= Set seeds for debugging =======
def set_seed(seed = 1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    setup_logger(seed)

## ======== Experimental set-up ==========
# Set variables below to run different experiments
def ablate_networks(reset_latent_policy, reset_latent_values, reset_latent_rwds, networks):
    # Reset latent policy
    if reset_latent_policy:
        networks.policy_net.apply(
            networks.reset_param
        )  # Try randomly initialising policy net, to simulate cerebellum damage

    # Reset latent values
    if reset_latent_values:
        networks.value_net.apply(networks.reset_param)

    if reset_latent_rwds:
        networks.rwd_net.apply(networks.reset_param)

    return networks

## ------ Define starting states for additional analysis ----
def get_starting_state(env, start = None):
    if start is not None:
        # I specifically selected states which are not ecounted during the optimal traject from the training starting state
        # ES: early state, MS: mid state, LS: late state
        if start == 0:
            init_state = (2, 2, 0)  # 7 moves away from goal
            file_indx = "ES"
        elif start == 1:
            init_state = (0, 0, 2)  # 3 moves away from goal for N=3
            file_indx = "MS"
        elif start == 2:
            init_state = (1, 2, 2)  # 1 move away
            file_indx = "LS"
        init_state_idx = env.states.index(init_state)
        env.init_state_idx = init_state_idx

    # Start from random state
    else:
        file_indx = "RandState"
    return file_indx

## ======== Run acting ==========
@torch.no_grad()
def get_results(env, start, networks, mcts, episode, n_mcts_simulations_range, temperature):
    data = []
    for n in n_mcts_simulations_range:
        errors = []
        mcts.n_simulations = n
        for ep in range(episode):
            # Initialise list to store game variables
            episode_state = []
            episode_action = []
            episode_rwd = []
            episode_piProb = []
            episode_rootQ = []

            if start is not None:
                # Reset from pre-defined initial state
                c_state = env.reset()
            else:
                # Start from an initial random state (apart from goal)
                c_state = env.random_reset()

            # Compute min n. moves from current random state
            min_n_moves = hanoi_solver(
                tuple(env.current_state())
            )  # pass state represt not in one-hot form

            done = False
            step = 0
            while not done:
                # Run MCTS to select the action
                action, pi_prob, rootNode_Q = mcts.run_mcts(
                    c_state, networks, temperature=temperature, deterministic=False
                )

                # Take a step in env based on MCTS action
                n_state, rwd, done, illegal_move = env.step(action)
                step += 1

                # Store variables for training
                # NOTE: not storing the last terminal state (don't think it is needed)
                episode_state.append(c_state)
                episode_action.append(action)
                episode_rwd.append(rwd)
                episode_piProb.append(pi_prob)
                episode_rootQ.append(rootNode_Q)

                # current state becomes next state
                c_state = n_state

            errors.append(step - min_n_moves)

        data.append([n, sum(errors) / len(errors)])

    logging.info(data)
    return data

## ===== Save results =========
def save_results_to_file(seed, file_indx, command_line, data, reset_latent_policy, reset_latent_values, reset_latent_rwds, save_results, timestamp):
    # Create directory to store results
    file_dir = os.path.join("stats", "Hanoi", timestamp)
    file_dir = os.path.join(file_dir, str(seed), str(file_indx))
    # Create directory if it did't exist before
    os.makedirs(file_dir, exist_ok=True)

    label_1, label_2, label_3 = "", "", ""
    if reset_latent_policy or reset_latent_values or reset_latent_rwds:
        if reset_latent_policy:
            label_1 = "ResetLatentPol_"
        if reset_latent_values:
            label_2 = "ResetLatentVal_"
        if reset_latent_rwds:
            label_3 = "ResetLatentRwd_"

    # Allow to combine reset rwd with other reset
    label = label_1 + label_2 + label_3

    if label == "":
        label = "Muzero_"

    acc_dir = os.path.join(file_dir, label + "actingAccuracy.pt")

    # Store command line
    if save_results:
        with open(os.path.join(file_dir, label + "commands.txt"), "w") as f:
            f.write(command_line)
        # Store accuracy
        torch.save(torch.tensor(data), acc_dir)

if __name__ == "__main__":
    # Run the script
    # env_name = "Hanoi"
    # N = 3
    # max_steps = 200
    # env = TowersOfHanoi(N=N, max_steps=max_steps)
    # s_space_size = env.oneH_s_size
    # n_action = 6  # n. of action available in each state for Tower of Hanoi (including illegal ones)
    # discount = 0.8
    # n_mcts_simulations_range = [
    #     5,
    #     10,
    #     30,
    #     50,
    #     80,
    #     110,
    #     150,
    # ]  # 25 during acting n. of mcts passes for each step

    # lr = 0
    # TD_return = True  # needed for NN to use logit to scalar transform
    # model_run_n = 1
    # save_results = True
    # seed_indx = s

    # episode = 100
    # dirichlet_alpha = 0
    # temperature = 0  # 0.1
    # discount = 0.8
    # 
    parser = argparse.ArgumentParser(description="Muzero acting ablation experiments")
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed for random number generation (default: 1)",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="Hanoi",
        help="Environment to train on (default: Hanoi)",
    )
    parser.add_argument(
        "--n_mcts_simulations_range",
        type=list,
        default=[5, 10, 30, 50, 80, 110, 150],
        help="Range of MCTS simulations to test (default: [5, 10, 30, 50, 80, 110, 150])",
    )
    parser.add_argument(
        "--model_run_n",
        type=int,
        default=1,
        help="Model run number (default: 1)",
    )
    parser.add_argument(
        "--save_results",
        type=bool,
        default=True,
        help="Save results (default: True)",
    )
    parser.add_argument(
        "--reset_latent_policy",
        type=bool,
        default=False,
        help="Reset latent policy (default: False)",
    )
    parser.add_argument(
        "--reset_latent_values",
        type=bool,
        default=False,
        help="Reset latent values (default: True)",
    )
    parser.add_argument(
        "--reset_latent_rwds",
        type=bool,
        default=False,
        help="Reset latent rewards (default: True)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=100,
        help="Number of episodes to run (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Temperature for MCTS (default: 0)",
    )
    parser.add_argument(
        "--dirichlet_alpha",
        type=float,
        default=0,
        help="Dirichlet alpha for exploration (default: 0)",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=0.8,
        help="Discount factor (default: 0.8)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Maximum number of steps per episode (default: 200)",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=3,
        help="Number of disks in the Towers of Hanoi (default: 3)",
    )
    parser.add_argument("--lr" , type=float, default=0, help="Learning rate (default: 0)")
    parser.add_argument(
        "--TD_return",
        type=bool,
        default=True,
        help="Use TD return (default: True)",
    )
    parser.add_argument("--n_action", type=int, default=6, help="Number of actions (default: 6)")
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Starting state index (default: None)",
    )
    parser.add_argument("--timestamp", type=str, default=None, help="Timestamp for results")
    args = parser.parse_args()
    seed = args.seed
    env_name = args.env_name
    n_mcts_simulations_range = args.n_mcts_simulations_range
    model_run_n = args.model_run_n
    save_results = args.save_results
    reset_latent_policy = args.reset_latent_policy
    reset_latent_values = args.reset_latent_values
    reset_latent_rwds = args.reset_latent_rwds
    episode = args.episode
    temperature = args.temperature
    dirichlet_alpha = args.dirichlet_alpha
    discount = args.discount
    max_steps = args.max_steps
    N = args.N
    lr = args.lr
    TD_return = args.TD_return
    n_action = args.n_action
    start = args.start
    timestamp = args.timestamp

    env = TowersOfHanoi(N=N, max_steps=max_steps)
    s_space_size = env.oneH_s_size


    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_indx = get_starting_state(env, start)

    ## ====== Log command line =====
    command_line = f"Seed: {seed}, Env: {env_name}, Run type: {file_indx}, Episodes: {episode},  Reset latent pol: {reset_latent_policy}, Reset latent value: {reset_latent_values}, Reset latent rwd: {reset_latent_rwds}, n. MCTS: {n_mcts_simulations_range}, N. disks: {N}"
    logging.info(command_line)

    # ======= Load pre-trained NN ========
    mcts = MCTS(
        discount=discount,
        root_dirichlet_alpha=dirichlet_alpha,
        n_simulations=n_mcts_simulations_range[0],
        batch_s=1,
        device=dev,
    )
    networks = MuZeroNet(
        rpr_input_s=s_space_size, action_s=n_action, lr=lr, TD_return=TD_return, device=dev
    ).to(dev)
    model_path = os.path.join("stats", "Hanoi", timestamp, "muzero_model.pt")
    model_dict = torch.load(
        model_path, map_location=dev
    )
    networks.load_state_dict(model_dict["Muzero_net"])
    networks.optimiser.load_state_dict(model_dict["Net_optim"])

    networks = ablate_networks(
        reset_latent_policy, reset_latent_values, reset_latent_rwds, networks
    )
    data = get_results(
        env,
        start,
        networks,
        mcts,
        episode,
        n_mcts_simulations_range,
        temperature,
    )

    save_results_to_file(
        seed,
        file_indx,
        command_line,
        data,
        reset_latent_policy,
        reset_latent_values,
        reset_latent_rwds,
        save_results,
        timestamp
    )
    logging.info("Results saved")