"""


"""
from numbers import Number
import copy
import random

import numpy as np
import torch
from torch import nn


def softmax(values, t=1.0):
    """
    Rescales inputs into range[0-1] to an accumulative sum of t.
    :param values: tensor of values to rescale
    :param t: softmax temperature - higher values result in a more uniformly-distributed output
    """
    return torch.softmax(values / t, dim=0)

    

def softmax_selection(values, t=1.0):
    """
    Returns a multinomial output of the values input tensor in the shape of (input rows x num_samples)
    """
    return torch.multinomial(softmax(values, t=t), 1).item()


class TransitionMemory:
    """
    Stores & manages state transitions of agent experiences 
    """

    def __init__(self, capacity, state_size, device):
        """
        A memory of state transitions
        :param capacity: size of the memory. (old transitions will start dropping out when the capacity is reached)
        :param state_size: int or tuple - the shape of the state vectors
        :param device: cpu or cuda
        """
        self.index = -1
        self.size = 0
        self.capacity = capacity
        if isinstance(state_size, Number):
            state_size = [state_size]
        self.states = torch.zeros((capacity, *state_size), device=device)
        self.actions = torch.zeros(capacity, device=device, dtype=torch.long)
        self.rewards = torch.zeros(capacity, device=device)
        self.new_states = torch.zeros((capacity, *state_size), device=device)
        self.done = torch.zeros(capacity, device=device)

    def add(self, state, action, reward, new_state, done):
        """
        Adds a new experience to memory.

        
        :param state: Current state of agent.
        :param action: Action taken relative to current state.
        :param reward: actual rewarded that action results in.
        :param new_state: New state as a consequence of action.
        :param done: Boolean identifies complete policy transition.
        """
        self.index = (self.index + 1) % self.capacity

        self.states[self.index, :] = torch.tensor(state)
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.new_states[self.index, :] = torch.tensor(new_state)
        self.done[self.index] = torch.tensor(float(done))

        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        """
        Returns a random sample of experiences

        :param n: number of experiences to return
        """

        n = min(n, self.size)

        idx = np.random.choice(self.size, n, replace=False)
        return (
            self.states[idx, :],
            self.actions[idx],
            self.rewards[idx],
            self.new_states[idx, :],
            self.done[idx],
        )

    def last(self):
        """
        Samples data off of last index returning final policy values
        
        """
        return (
            self.states[self.index, :],
            self.actions[self.index],
            self.rewards[self.index],
            self.new_states[self.index],
            self.done[self.index],
        )


class DQN:
    """
    A deep Q network that optionally implements the rule from our paper:
     Brain-Inspired modulation of reward-prediction error improves reinforcement learning adaptation to environmental
     change
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
        self,
        network: nn.Sequential,
        input_shape,
        batch_size,
        replay_buffer_size,
        update_frequency=10,
        lr=1e-3,
        sync_frequency=5,
        gamma=0.95,
        epsilon=0.1,
        softmax_temp=None,
        seed=42,
        weight_decay=0,
        loss_fn: torch.nn.Module | None = None,
    ):
        """
        :param network: deep network, of type torch.nn.Sequential
        :param input_shape: shape of input vectors for the network
        :param batch_size: number of experiences to sample from the replay buffer at each learning step
        :param replay_buffer_size: number of experiences to keep in the replay buffer
        :param update_frequency: number of steps before updating models
        :param lr: learning rate
        :param sync_frequency: number of steps to run between syncing the policy and value networks
        :param gamma: discount factor
        :param epsilon: parameter for e-greedy action sampling
        :param softmax_temp: softmax temperature for use in new RL rule
        :param seed: random seed
        :param weight_decay: rate at which connection between nodes decays to zero (l2 regularization)
        """
        torch.manual_seed(seed)

        self.policy_net = copy.deepcopy(network)
        self.value_net = copy.deepcopy(network)

        self.policy_net.to(self.device)
        self.value_net.to(self.device)

        # instantiate loss and optimizer
        self.loss_fn: torch.nn.Module = loss_fn or torch.nn.MSELoss()
        self.lr = lr
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=lr, weight_decay=weight_decay
        )

        # instantiate experience memory
        self.transition_memory = TransitionMemory(
            capacity=replay_buffer_size, state_size=input_shape, device=self.device
        )

        # store other params
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.n_outputs = self.policy_net[-1].out_features
        self.sync_frequency = sync_frequency
        self.sync_counter = 0
        self.gamma = torch.tensor(gamma, device=self.device)
        self.epsilon = epsilon
        self.update_counter = 0
        self.softmax_temp = softmax_temp

    def get_action_values(self, state):
        """
        Fetches value of future env state and expands array shape.

        :param state: Current state of agent.
        """
        with torch.no_grad():
            return self.policy_net(
                torch.tensor(
                    np.expand_dims(state, 0).astype(np.float32), device=self.device
                )
            )[0]

    def select_action(self, state):
        """
        Creates parameters for random action or induces action based on policy values.

        :param state: Current state of agent.
        """
        if self.softmax_temp is None and random.random() < self.epsilon:
            return np.random.choice(self.n_outputs)

        with torch.no_grad():
            values = self.policy_net(
                torch.tensor(
                    np.expand_dims(state, 0).astype(np.float32), device=self.device
                )
            )[0]

        if self.softmax_temp is not None:
            return softmax_selection(values, self.softmax_temp)

        max_q, index = values.max(0)
        return index.item()

    def update(self, state, action, reward, new_state, done):
        """
        Updates policy values upon a new agent state.  

        :param state: Current state of agent.
        :param action: Action taken relative to current state.
        :param reward: Actual reward agent action results in.
        :param new_state: New state as a consequence of action.
        :param done: Boolean identifies complete policy transition.
        """

        self.transition_memory.add(state, action, reward, new_state, done)
        self.update_counter += 1
        if self.update_counter % self.update_frequency == 0:

            # sync value and policy networks
            self.sync_counter += 1
            if self.sync_counter % self.sync_frequency == 0:
                self.value_net.load_state_dict(self.policy_net.state_dict())

            s, a, r, ns, d = self.transition_memory.sample(self.batch_size)

            # get policy network's current value estimates
            state_action_values = self.policy_net(s)

            # get target value estimates, based on actual rewards and value net's predictions of next-state value
            with torch.no_grad():
                new_state_value, _ = self.value_net(ns).max(1)

            target_action_value = r + self.gamma * new_state_value * (1 - d)
            target_values = state_action_values.clone().detach()
            target_values[np.arange(target_values.shape[0]), a] = target_action_value

            # optimize loss
            loss = self.loss_fn(state_action_values, target_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()

    def update_single(self, state, action, reward, new_state, done):
        """
        Not sure what this does, does not seem to be in use. >.< 

        :param state: Current state of agent.
        :param action: Action taken relative to current state.
        :param reward: Actual reward agent action results in.
        :param new_state: New state as a consequence of action.
        :param done: Boolean identifies complete policy transition.
        """

        state = torch.tensor(
            np.expand_dims(state, 0).astype(np.float32), device=self.device
        )
        new_state = torch.tensor(
            np.expand_dims(new_state, 0).astype(np.float32), device=self.device
        )

        # get policy network's current value estimates
        state_action_values = self.policy_net(state)

        # get target value estimates, based on actual rewards and value net's predictions of next-state value
        with torch.no_grad():
            new_state_value, _ = self.value_net(new_state).max(1)
        target_action_value = reward + self.gamma * new_state_value * (1 - done)
        target_values = state_action_values.clone().detach()
        target_values[np.arange(target_values.shape[0]), action] = target_action_value

        # optimize loss
        loss = self.loss_fn(state_action_values, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
