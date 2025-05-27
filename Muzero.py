import logging

import numpy as np
import torch
import torch.nn.functional as F

from buffer import Buffer
from MCTS.mcts import MCTS
from networks import MuZeroNet
from utils import adjust_temperature, compute_MCreturns, compute_n_step_returns
from torch.cuda.amp import autocast, GradScaler

class Muzero:

    def __init__(
        self,
        env,
        s_space_size,
        n_action,
        discount,
        dirichlet_alpha,
        n_mcts_simulations,
        unroll_n_steps,
        batch_s,
        TD_return,
        n_TD_step,
        lr,
        buffer_size,
        priority_replay,
        device,
        n_ep_x_loop=1,
        n_update_x_loop=1,
    ):

        self.dev = device

        ## ========= Set env variables========
        self.env = env
        self.n_ep_x_loop = (
            n_ep_x_loop  # how many env eps we collect for each training loop
        )
        self.discount = discount
        self.n_action = n_action

        ## ======= Set MuZero training variables =======
        self.unroll_n_steps = unroll_n_steps
        self.n_update_x_loop = (
            n_update_x_loop  # Muzero training steps x n. of training loop
        )
        self.batch_s = batch_s
        self.TD_return = TD_return
        self.n_step = n_TD_step

        ## ========== Initialise MuZero components =======
        self.mcts = MCTS(
            discount=self.discount,
            root_dirichlet_alpha=dirichlet_alpha,
            n_simulations=n_mcts_simulations,
            batch_s=batch_s,
            device=self.dev,
        )

        self.networks = MuZeroNet(
            rpr_input_s=s_space_size,
            action_s=self.n_action,
            lr=lr,
            TD_return=TD_return,
            device=self.dev,
        ).to(self.dev)

        self.networks = torch.compile(self.networks, mode="reduce-overhead")

        ## ========== Initialise buffer ========
        self.buffer = Buffer(
            buffer_size,
            unroll_n_steps,
            d_state=s_space_size,
            n_action=self.n_action,
            device=self.dev,
        )
        self.priority_replay = priority_replay

    def training_loop(self, n_loops, min_replay_size, print_acc=5, profiler=None):

        logging.info("Training started \n")

        accuracy = []  # in terms of mean n. steps to solve task
        tot_accuracy = []
        value_loss, rwd_loss, pi_loss = [], [], []
        cumulative_success = []
        total_success = 0
        q_values_per_loop = []
        grad_norms = []

        all_value_loss, all_rwd_loss, all_pi_loss = [], [], []

        state_action_history = {}

        for n in range(1, n_loops):
            ep_accuracy = []
            successes = 0
            q_values_this_loop = []
            for ep in range(self.n_ep_x_loop):
                # Play one episode
                steps, states, rwds, actions, pi_probs, returns, priorities, state_action_pairs = (
                    self._play_game(episode=n * self.n_ep_x_loop, deterministic=False)
                )

                for state_key, action in state_action_pairs:
                    if state_key not in state_action_history:
                        state_action_history[state_key] = []
                    state_action_history[state_key].append(action)

                # q_values_this_loop.append(mean_q_value)
                ep_accuracy.append(steps)
                # Store episode in buffer only if successful
                if returns[-1, 0] > 0:
                    successes += 1
                    self.buffer.add(
                        states, rwds, actions, pi_probs, returns, priorities
                    )
            accuracy.append(sum(ep_accuracy) / self.n_ep_x_loop)
            total_success += successes
            cumulative_success.append(total_success)

            # mean_q_value_loop = float(np.mean(q_values_this_loop)) if q_values_this_loop else 0.0
            # q_values_per_loop.append(mean_q_value_loop)

            # If time to train, train MuZero network
            if self.buffer.__len__() > min_replay_size:
                for t in range(self.n_update_x_loop):
                    if self.priority_replay:
                        (
                            states,
                            rwds,
                            actions,
                            pi_probs,
                            returns,
                            priority_indx,
                            priority_w,
                        ) = self.buffer.priority_sample(self.batch_s)
                    else:
                        states, rwds, actions, pi_probs, returns = (
                            self.buffer.uniform_sample(self.batch_s)
                        )
                        priority_w, priority_indx = None, None

                    # Update network
                    new_priority_w, v_loss, r_loss, p_loss, grad_norm = self._update(
                        states, rwds, actions, pi_probs, returns, priority_w
                    )
                    # Update buffer priorities
                    self.buffer.update_priorities(priority_indx, new_priority_w)
                    grad_norms.append(grad_norm)

                value_loss.append(v_loss)
                rwd_loss.append(r_loss)
                pi_loss.append(p_loss)

            if n * self.n_ep_x_loop % print_acc == 0:
                mean_acc = sum(accuracy) / print_acc
                mean_v_loss = sum(value_loss)/print_acc
                mean_r_loss = sum(rwd_loss)/print_acc
                mean_p_loss = sum(pi_loss)/print_acc
                logging.info(f"| Training Loop: {n} ")
                logging.info(f"Number of steps:  {mean_acc}")
                logging.info(f"V loss:  {mean_v_loss}")
                logging.info(f"rwd loss:  {mean_r_loss}")
                logging.info(f"Pi loss:  {mean_p_loss} \n")
                tot_accuracy.append(mean_acc)
                all_value_loss.append(mean_v_loss)
                all_rwd_loss.append(mean_r_loss)
                all_pi_loss.append(mean_p_loss)
                accuracy = []
                value_loss, rwd_loss, pi_loss = [], [], []
            if profiler is not None:
                profiler.step()

        # Compute average decisions per state
        avg_decisions_per_state = {}
        for state_key, actions in state_action_history.items():
            avg_decisions_per_state[state_key] = np.mean(actions)

        return {
            "tot_accuracy": tot_accuracy,
            "all_value_loss": all_value_loss,
            "all_rwd_loss": all_rwd_loss,
            "all_pi_loss": all_pi_loss,
            "cumulative_success": cumulative_success,
            "grad_norms": grad_norms,
            "state_action_history": state_action_history,
            "avg_decisions_per_state": avg_decisions_per_state,
        }

    def _play_game(self, episode, deterministic=False):

        # Initialise list to store game variables
        episode_state = []
        episode_action = []
        episode_rwd = []
        episode_piProb = []
        episode_rootQ = []

        c_state = self.env.reset()
        done = False
        step = 0

        sim_counts = []
        avg_visit_counts = []

        state_action_pairs = []

        while not done:
            # Run MCTS to select the action
            action, pi_prob, rootNode_Q = self.mcts.run_mcts(
                c_state,
                self.networks,
                temperature=adjust_temperature(episode),
                deterministic=deterministic,
            )

            state_key = self._state_to_key(c_state)
            state_action_pairs.append((state_key, action))

            # Take a step in env based on MCTS action
            n_state, rwd, done, _ = self.env.step(action)
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

        # TODO return this mean_q_value = float(mean(episode_rootQ)) if episode_rootQ else 0.0

        # Compute appropriate returns for each state
        if self.TD_return:
            episode_returns = compute_n_step_returns(
                episode_rwd, episode_rootQ, self.n_step, self.discount
            )
        else:
            episode_returns = compute_MCreturns(episode_rwd, self.discount)

        # Compute priorities for buffer
        priorities = torch.abs(
            torch.tensor(episode_returns) - torch.tensor(episode_rootQ)
        )

        # Organise ep. trajectory into appropriate transitions for training - i.e. each transition should have unroll_n_steps associated transitions for training
        states, rwds, actions, pi_probs, returns = self.organise_transitions(
            episode_state, episode_rwd, episode_action, episode_piProb, episode_returns
        )

        return step, states, rwds, actions, pi_probs, returns, priorities, state_action_pairs

    def _state_to_key(self, state):
        """Convert state to a hashable key for tracking"""
        if isinstance(state, torch.Tensor):
            return tuple(state.cpu().numpy().flatten())
        elif isinstance(state, np.ndarray):
            return tuple(state.flatten())
        else:
            return str(state)

    def _update(self, states, rwds, actions, pi_probs, returns, priority_w):
        # TRIAL: expands all states (including final ones) of mcts_steps for simplicty for steps after terminal just map everything to zero
        # if does not work then need to adapt for terminal states not to expand tree of mcts_steps

        rwd_loss, value_loss, policy_loss = (0, 0, 0)

        h_states = self.networks.represent(states)

        tot_pred_values = []

        for t in range(self.unroll_n_steps):

            pred_pi_logits, pred_values = self.networks.prediction(h_states)

            # Convert action to 1-hot encoding
            oneH_action = (
                torch.nn.functional.one_hot(
                    actions[:, t], num_classes=self.networks.num_actions
                )
                .squeeze()
                .to(self.dev, dtype=torch.long)
            )
            # oneH_action = actions[:,t].squeeze().to(self.dev)
            h_states, pred_rwds = self.networks.dynamics(h_states, oneH_action)

            # Scale the gradient for dynamics function by 0.5.
            h_states.register_hook(lambda grad: grad * 0.5)

            value_loss += F.mse_loss(
                pred_values.squeeze(), returns[:, t], reduction="none"
            )
            rwd_loss += F.mse_loss(pred_rwds.squeeze(), rwds[:, t], reduction="none")
            # NOTE: F.cross_entropy takes input as logits, and compute softmax inside the function
            policy_loss += F.cross_entropy(
                pred_pi_logits, pi_probs[:, t], reduction="none"
            )

            tot_pred_values.append(pred_values)

        loss = value_loss + rwd_loss + policy_loss

        new_priorities = None  # predefine new priorities in case no priority buffer

        if priority_w is not None:
            # Scale loss using importance sampling weights (based on priorty sampling from buffer)
            loss = loss * priority_w.detach()
            # Compute new priorities to update priority buffer
            with torch.no_grad():
                tot_pred_values = torch.stack(tot_pred_values, dim=1).squeeze(-1)
                new_priorities = (
                    (tot_pred_values[:, 0] - returns[:, 0]).abs().cpu().numpy()
                )
                # TODO fix cpu

        loss = loss.mean()
        # Scale the loss by 1/unroll_steps.
        loss.register_hook(lambda grad: grad * (1 / self.unroll_n_steps))

        # Update network
        self.networks.update(loss)

        # ---- Compute gradient norm ----
        total_norm = 0.0
        for p in self.networks.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = total_norm ** 0.5
        # --------------------------------

        return (
            new_priorities,
            value_loss.mean().detach(),
            rwd_loss.mean().detach(),
            policy_loss.mean().detach(),
            grad_norm
        )

    def organise_transitions(
        self,
        episode_state,
        episode_rwd,
        episode_action,
        episode_piProb,
        episode_returns,
    ):
        """Orgnise transitions in appropriate format, each state is associated to the n_step target values (pi_probs, rwds, MC_returs) for unroll_n_steps
        Returns:
            pi_probs: np.array(n_steps,unroll_n_steps,d_action)
            all others: np.array(n_steps,unroll_n_steps)
        """

        ## ===========================================
        # Try to remove unroll_n_steps terminal states, NOTE: Not a permanent solution, need terminal states to solve Hanoi with as few moves as possible
        # episode_state = episode_state[:-self.unroll_n_steps] # REMOVE this line, just to see if learning problem is driven by terminal states
        ## ===========================================

        n_states = len(episode_state)

        # Add "padding" for terminal states, which don't have enough unroll_n_steps ahead
        # by trating states over the end of game as absorbing states
        # NOTE: This can cause issues, since a lot of cycles in Hanoi and zero is a real action
        episode_rwd += [0] * self.unroll_n_steps
        # episode_action += [0] * self.unroll_n_steps
        episode_action += [
            np.random.randint(0, self.n_action)
        ] * self.unroll_n_steps  # select uniform random action for unroll_n_steps over the end
        episode_returns += [0] * self.unroll_n_steps
        absorbing_policy = torch.ones_like(episode_piProb[-1]) / episode_piProb[-1].numel()
        episode_piProb += [absorbing_policy] * self.unroll_n_steps

        # Initialise variables for storage
        rwds = torch.zeros((n_states, self.unroll_n_steps), dtype=torch.float32, device=self.dev)
        actions = torch.zeros((n_states, self.unroll_n_steps), dtype=torch.long, device=self.dev)
        pi_probs = torch.zeros((n_states, self.unroll_n_steps, len(episode_piProb[0])), dtype=torch.float32, device=self.dev)   
        returns = torch.zeros((n_states, self.unroll_n_steps), dtype=torch.float32, device=self.dev)

        for i in range(n_states):
            rwds[i, :] = torch.tensor(episode_rwd[i : i + self.unroll_n_steps], device=self.dev)
            actions[i, :] = torch.tensor(episode_action[i : i + self.unroll_n_steps], device=self.dev)
            pi_probs[i, :, :] = torch.stack([p.to(self.dev) for p in episode_piProb[i : i + self.unroll_n_steps]], dim=0)
            returns[i, :] = torch.tensor(episode_returns[i : i + self.unroll_n_steps], dtype=torch.float32, device=self.dev)

        # TODO return np.array(episode_state), rwds, actions, pi_probs, returns
        device = self.dev
        states = torch.stack([torch.tensor(s, dtype=torch.float32, device=self.dev) for s in episode_state])
        # states = torch.tensor(np.array(episode_state), dtype=torch.float32, device=device)
        # TODO rwds = torch.tensor(rwds, dtype=torch.float32, device=device)
        # actions = torch.tensor(actions, dtype=torch.long, device=device)
        # pi_probs = torch.tensor(pi_probs, dtype=torch.float32, device=device)
        # returns = torch.tensor(returns, dtype=torch.float32, device=device)
        return states, rwds, actions, pi_probs, returns
