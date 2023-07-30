import pickle
import os

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.agents.dqn import DQN

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
import pandas as pd



from minigrid.core.constants import OBJECT_TO_IDX
object_vector_map = {
    OBJECT_TO_IDX['wall']: [1, 0, 0],
    OBJECT_TO_IDX['goal']: [0, 1, 0],
    OBJECT_TO_IDX['goalNoTerminate']: [0, 0, 1],
}


class Image2VecWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        """
        create a new 3x7x7 state vector out of the image the env returns:
        vector[i, j, 0] is 1 if the object at (i,j) is a wall
        vector[i, j, 1] is 1 if the object at (i,j) is a goal
        vector[i, j, 2] is 1 if the object at (i,j) is a transient goal
        :param image: image array supplied by the TransientGoals env
        :return: a new vector as described above
        """
        image = observation['image']
        vec = np.zeros(image.shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                vec[i, j, :] = object_vector_map.get(image[i, j, 0], [0, 0, 0])
        return np.moveaxis(vec, (2, 0, 1), (0, 1, 2))


def run_trial(dropout, trial_id=None):
    trial_id = trial_id or os.getpid()

    # instantiate environment
    env = Image2VecWrapper(TransientGoals(tile_size=32, render_mode='none'))

    # a multi-layer network
    multilayer = nn.Sequential(
        nn.Flatten(1, -1),
        nn.Dropout(p=(dropout / 100)),
        nn.Linear(147, 10),
        nn.Tanh(),
        nn.Dropout(p=(dropout / 100)),
        nn.Linear(10, 3)
    )

    # a convolutional network - seems to give different behavior
    # conv = nn.Sequential(
    #     nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1),
    #     nn.ReLU(inplace=True),
    #     nn.BatchNorm2d(2),
    #     # nn.AvgPool2d(kernel_size=3, stride=2),
    #     # nn.MaxPool2d(kernel_size=3, stride=2),
    #     nn.Flatten(1, -1),
    #     nn.Dropout(p=(dropout / 100)),
    #     nn.Linear(98, 3),
    # )

    agent = DQN(
        network=multilayer,  # supply either network here
        input_shape=(3, 7, 7),
        replay_buffer_size=10000,
        update_frequency=5,
        lr=0.005,
        sync_frequency=25,
        gamma=0.85, epsilon=0.05,
        batch_size=1500
    )

    # run a number of steps in the environment
    N_STEPS = 30000
    episode_num = 0
    cumulative_reward = 0

    # setup results dataframe
    results = pd.DataFrame()

    # create the environment
    state, _ = env.reset()

    for step in range(N_STEPS):
        # get & execute action
        action = agent.select_action(np.expand_dims(state, 0))
        new_state, reward, done, _, _ = env.step(action)

        # use this experience to update agent
        agent.update(state, action, reward, new_state, done=False)

        # reset environment if done (ideally env would do this itself)
        if done:
            print(str(dropout) + f' goal reached at step {step}/{N_STEPS}')
            state, _ = env.reset()
            episode_num += 1
        else:
            state = new_state

        # add results to the history
        transient_goal = reward == env.transient_reward
        terminal_goal = reward == env.termination_reward
        cumulative_reward += reward
        results = pd.concat(
            (
                results,
                pd.DataFrame([
                    {
                        'trial_id': trial_id,
                        'dropout': dropout,
                        'step': step,
                        'episode': episode_num,
                        'reward': reward,
                        'cumulative_reward': cumulative_reward,
                        'terminal_goal': terminal_goal,
                        'transient_goal': transient_goal
                    }])
            ),
            ignore_index=True
        )

    return results, agent


def run_trial_and_save(dropout, filename, trial_id=None):
    results, agent = run_trial(dropout, trial_id)
    with open(filename, 'wb') as f:
        pickle.dump(
            {'results': results, 'agent': agent},
            f
        )
    return results, agent


def load_trial(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['results'], data['agent']


if __name__ == '__main__':
    results_healthy, agent_healthy = run_trial_and_save(dropout=0, trial_id=1, filename='0.pkl')
    results_dep, agent_dep = run_trial_and_save(dropout=66, trial_id=2, filename='66.pkl')
