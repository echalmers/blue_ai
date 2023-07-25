import argparse

from blue_ai_envs.envs.transient_goals import TransientGoals
from blue_ai_envs.agents.dqn import DQN

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from torch import nn



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


def train(dropout):
    # instantiate environment
    env = Image2VecWrapper(gym.make('blue_ai_envs/TransientGoals', tile_size=32, render_mode='none'))

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
    reward_history = np.zeros(N_STEPS)

    # create the environment
    state, _ = env.reset()
    print(state)

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
        else:
            state = new_state

        # add reward to the history
        reward_history[step] = reward

    return reward_history


if __name__ == '__main__':
    plt.plot(train(dropout=0).cumsum())
    plt.plot(train(dropout=50).cumsum())
    plt.legend(['0% dropout (healthy)', '50% dropout (depressed)'])
    plt.title('cumulative reward')
    plt.show()
