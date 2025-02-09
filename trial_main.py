import numpy as np
import torch

from env.hanoi import TowersOfHanoi
from MCTS.mcts import MCTS
from networks import MuZeroNet

## ========= Useful variables: ===========
discount = 0.99
dirichlet_alpha = 0.03
temperature = 1
n_simulations = 10
batch_s = 1
lr = 0.001
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(dev)

## ========= Initialise env ========
N = 3
max_steps = 200
env = TowersOfHanoi(N, max_steps)

a_space_s = (
    6  # 6 actions available (including iligal) for any state and number of disks
)
state_space = env.states
s_space_s = len(state_space)

## We assume a 1-to-1 correspondance between 1hot representations and the order of the original states in the tower of Hanoi
## namely, the vector [1,0,0,...] refers to the first generated state in the tower and so on
one_hot_s = np.eye(
    s_space_s
)  # this creates a matrix whose columns represent a different 1hot vector for each state
## =================================

mcts = MCTS(
    discount=discount,
    root_dirichlet_alpha=dirichlet_alpha,
    n_simulations=n_simulations,
    batch_s=batch_s,
    #  lr=lr,
    device=dev,
)
networks = MuZeroNet(rpr_input_s=s_space_s, action_s=a_space_s, lr=lr, device=dev)

c_state = env.reset()
c_s_indx = env.init_state_idx
done = False

step = 0
while not done:

    oneH_c_s = one_hot_s[:, c_s_indx]

    # Run MCTS to select the action
    action, pi_prob, rootNode_Q = mcts.run_mcts(
        oneH_c_s, networks, temperature, deterministic=False
    )
    print(step)
    print(action)

    n_state, rwd, done, illegal_move = env.step(action)
    step += 1

    print("illegal: ", illegal_move)
    print(n_state, "\n")

    if step == max_steps:
        done = True

    if not done:
        print(state_space, n_state)
        n_s_indx = state_space.index(n_state)  # Compute new c_state index if not done

    # current state becomes next state
    c_s_indx = n_s_indx
