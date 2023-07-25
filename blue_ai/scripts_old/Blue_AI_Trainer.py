# Blue_AI_Trainer.py
# Jesse Viehweger, Eric Chalmers
# April 20, 2023

# Purpose:
# This code is used to train agents in the blue_ai_envs. Users are given the option
# the select the amount of dropout in the network provided the model is not being retrained.
# Users can either train a new model, or if a agent file with the appropriate name
# (ex. agent20.pkl) is located within the Agents directory it will further train that agent.
# Example Call: python3 Blue_AI_Trainer.py --drop 20

# Results:
# Once the training has processed two new files may be generated for the user. The first 
# file is potentially a new agent file located in the agent directory. The second file
# is the recorded logs from the training session. It will save a file with a name coresponding
# to the dropout requested in the run (ex. log20.pkl) in the Logs direcotry. The log stores a a dictionary
# that stores 4 different values every step in the evironment.
# 1) optional_rewards: An integer list where each element indicates if a optional was optained that step.
# 2) mandatory_rewards: An integer list where each element indicates if a mandatory reward was optained that step.
# 3) num_frames_per_episode: An integer list that stores the current step count.
# 4) reward_per_episode: A float list that stores the reward optained from that step.

# Assumptions:
# If a network is being retrained the dropout value used with that agent still needs to be given
# as input for the data to be logged correctly.


from pathlib import Path
import torch
from torch import nn
import numpy as np
import gymnasium as gym
from blue_ai_envs.envs.transient_goals import TransientGoals
import random
import matplotlib.pyplot as plt
import copy
from numbers import Number
import pickle
import argparse
import os


class TransitionMemory:
    """
    A memory of state transitions
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
        self.index = (self.index + 1) % self.capacity

        self.states[self.index, :] = torch.tensor(state)
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.new_states[self.index, :] = torch.tensor(new_state)
        self.done[self.index] = torch.tensor(float(done))

        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        n = min(n, self.size)

        idx = np.random.choice(self.size, n, replace=False)
        return self.states[idx, :], self.actions[idx], self.rewards[idx], self.new_states[idx, :], self.done[idx]

    def last(self):
        return self.states[self.index, :], self.actions[self.index], self.rewards[self.index], self.new_states[self.index], self.done[self.index]


class DQN:
    """
    A deep Q network that optionally implements the rule from our paper:
     Brain-Inspired modulation of reward-prediction error improves reinforcement learning adaptation to environmental
     change
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, network: nn.Sequential, input_shape, batch_size, replay_buffer_size, update_frequency=10,
                 lr=1e-3, sync_frequency=5,
                 gamma=0.95, epsilon=0.1,
                 softmax_temp=1.0,
                 seed=42):
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
        """
        torch.manual_seed(seed)

        self.policy_net = copy.deepcopy(network)
        self.value_net = copy.deepcopy(network)
        self.policy_net.to(self.device)
        self.value_net.to(self.device)

        # instantiate loss and optimizer
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        # instantiate experience memory
        self.transition_memory = TransitionMemory(capacity=replay_buffer_size, state_size=input_shape, device=self.device)

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

    def select_action(self, state):
        if random.random() < self.epsilon:
            return np.random.choice(self.n_outputs)
        with torch.no_grad():
            max_q, index = self.policy_net(torch.tensor(state.astype(np.float32), device=self.device))[0].max(0)
        return index.item()

    def update(self, state, action, reward, new_state, done):

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


from minigrid.core.constants import OBJECT_TO_IDX
object_vector_map = {
    OBJECT_TO_IDX['wall']: [1, 0, 0],
    OBJECT_TO_IDX['goal']: [0, 1, 0],
    OBJECT_TO_IDX['goalNoTerminate']: [0, 0, 1],
}
def image_to_vec(image):
    """
    create a new 3x7x7 state vector out of the image the env returns:
    vector[i, j, 0] is 1 if the object at (i,j) is a wall
    vector[i, j, 1] is 1 if the object at (i,j) is a goal
    vector[i, j, 2] is 1 if the object at (i,j) is a transient goal
    :param image: image array supplied by the TransientGoals env
    :return: a new vector as described above
    """
    vec = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            vec[i, j, :] = object_vector_map.get(image[i, j, 0], [0, 0, 0])
    return np.moveaxis(vec, (2, 0, 1), (0, 1, 2))

#main trainer code

if __name__ == '__main__':

    #get desired dropout
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drop", help="amount of dropout to be used in the model", default=0
    )
    args = parser.parse_args()
    drop = args.drop
    drop = int(drop)

    # instantiate environment
    env = gym.make('blue_ai_envs/TransientGoals', tile_size=32)

    # a multi-layer network
    multilayer = nn.Sequential(
        nn.Flatten(1, -1),
        nn.Dropout(p=(drop/100)),
        nn.Linear(147, 10),
        nn.Tanh(),
        nn.Dropout(p=(drop/100)),
        nn.Linear(10, 3)
    )

    # a convolutional network - seems to give different behavior
    conv = nn.Sequential(
        nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(2),
        # nn.AvgPool2d(kernel_size=3, stride=2),
        # nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Flatten(1, -1),
        nn.Dropout(p=(drop/100)),
        nn.Linear(98, 3),
    )

    # instantiate DQN
    file1 = os.path.join('Agents', 'agent' + str(drop) +'.pkl')
    path1 = Path(file1)
    
    if(path1.is_file()):
        infile = open(file1,'rb')
        agent = pickle.load(infile)
        infile.close()
        print("Reusing Trained Network")
    else:
        agent = DQN(
        network=multilayer,   # supply either network here
        input_shape=(3, 7, 7),
        replay_buffer_size=10000,
        update_frequency=5,
        lr=0.005,
        sync_frequency=25,
        gamma=0.85, epsilon=0.05,
        batch_size=1500
        ) 
    
    logs = {"num_frames_per_episode": [], "reward_per_episode": [],
    "optional_rewards": [], "mandatory_rewards": []}
    greenNum = 0
    blueNum = 0

    # run a number of steps in the environment
    N_STEPS = 40000
    reward_history = np.zeros(N_STEPS)

    # create the environment
    state = env.reset()
    state = image_to_vec(state[0]['image'])

    for step in range(N_STEPS):
        # get & execute action chosen by DQN agent
        action = agent.select_action(np.expand_dims(state, 0))
        new_state, reward, done, _, _ = env.step(action)
        new_state = image_to_vec(new_state['image'])

        if (reward != 0):
            if((reward*100)%2 == 1):
                greenNum = greenNum + 1
            else:

                blueNum = blueNum + 1

        # use this experience to update agent
        agent.update(state, action, reward, new_state, done=False)

        # reset environment if done (ideally env would do this itself)
        if done:
            print(str(drop) + f' goal reached at step {step}/{N_STEPS}')
            state = env.reset()
            state = image_to_vec(state[0]['image'])
        else:
            state = new_state

        # add reward to the history
        reward_history[step] = reward

        logs["optional_rewards"].append(blueNum)
        logs["mandatory_rewards"].append(greenNum)
        logs["num_frames_per_episode"].append(step)
        logs["reward_per_episode"].append(reward)


    # Save the agent and logs
    if not os.path.exists('Agents'):
        os.makedirs('Agents')
    
    new_file_path = os.path.join('Agents', 'agent' + str(drop) +'.pkl')

    with open(new_file_path, 'wb') as f:
        pickle.dump(agent, f)

    if not os.path.exists('Logs'):
        os.makedirs('Logs')

    new_file_path = os.path.join('Logs', 'Log' + str(drop) +'.pkl')

    with open(new_file_path, 'wb') as f:
        pickle.dump(logs, f)

    input('press enter to close...')
