import pickle
import os

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.agents.dqn import DQN

import torch
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


class StaticDropout(nn.Module):
    def __init__(self, in_features, p):
        super().__init__()
        self._p = p
        self.in_features = in_features
        self.mask = torch.rand(in_features, requires_grad=False) > self._p

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        self._p = value
        self.mask = torch.rand(self.in_features, requires_grad=False) > self._p

    def forward(self, x):
        return x * self.mask


class ConnectionDropout(nn.Linear):

    def __init__(self, in_features: int, out_features: int, p: float):
        super().__init__(in_features, out_features, bias=True)
        self.p = p

    def forward(self, input):
        return nn.functional.linear(input, nn.functional.dropout(self.weight, p=self.p), self.bias)


def run_trial(dropout, trial_id=None, transient_reward=0.25, termination_reward=1.0, allow_done_action=True, callbacks=None, steps=30000):
    trial_id = trial_id or os.getpid()
    callbacks = callbacks or []

    # instantiate environment
    env = Image2VecWrapper(TransientGoals(render_mode='none', transient_reward=transient_reward, termination_reward=termination_reward))

    # a multi-layer network
    outputs = 4 if allow_done_action else 3
    multilayer = nn.Sequential(
        nn.Flatten(1, -1),
        nn.Dropout(p=(dropout / 100)),
        # LostSpinesLayer(in_features=147, p=dropout / 100),
        nn.Linear(147, 10),
        # ConnectionDropout(147, 10, p=dropout / 100),
        nn.Tanh(),
        nn.Dropout(p=(dropout / 100)),
        # StaticDropout(in_features=10, p=dropout / 100),
        nn.Linear(10, outputs)
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
        gamma=0.9, epsilon=0.05,
        batch_size=1500
    )

    # run a number of steps in the environment
    steps_this_episode = 0
    episode_num = 0
    cumulative_reward = 0

    # setup results dataframe
    results = pd.DataFrame()

    # create the environment
    state, _ = env.reset()

    for step in range(steps):
        steps_this_episode += 1

        for callback in callbacks:
            callback(agent=agent, step=step)

        # get & execute action
        action = agent.select_action(np.expand_dims(state, 0))
        new_state, reward, done, _, _ = env.step(action)

        # use this experience to update agent
        agent.update(state, action, reward, new_state, done=False)

        # reset environment if done (ideally env would do this itself)
        if done or steps_this_episode > 500:
            print(str(dropout) + f' goal reached at step {step}/{steps}' if done else '***TIMEOUT***')
            state, _ = env.reset()
            episode_num += 1
            steps_this_episode = 0
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

    return results, agent, env


class TrialRunner:

    def __init__(self, dropout, filename, trial_id=None, termination_reward=1.0, transient_reward=0.25, allow_done_action=True, callbacks=None, steps=30000):
        self.dropout = dropout
        self.filename = filename
        self.trial_id = trial_id
        self.transient_reward = transient_reward
        self.termination_reward = termination_reward
        self.allow_done_action = allow_done_action
        self.callbacks = callbacks
        self.steps = steps

    def __call__(self):
        results, agent, env = run_trial(
            dropout=self.dropout,
            trial_id=self.trial_id,
            termination_reward=self.termination_reward,
            transient_reward=self.transient_reward,
            allow_done_action=self.allow_done_action,
            callbacks=self.callbacks,
            steps=self.steps
        )
        with open(self.filename, 'wb') as f:
            pickle.dump(
                {'results': results, 'agent': agent, 'env': env},
                f
            )
        return results, agent


def load_trial(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['results'], data['agent'], data['env']


if __name__ == '__main__':
    import random

    for dropout in [0, 50]:
        for trial in range(10):
            for allow_done in [True, False]:

                TrialRunner(
                    dropout=dropout,
                    filename=os.path.join('.', 'data', f'highterminal_{dropout}_{trial}_{"done_allowed" if allow_done else ""}.pkl'),
                    trial_id=f'{dropout}-{trial}',
                    allow_done_action=allow_done
                )()

                TrialRunner(
                    dropout=dropout,
                    filename=os.path.join('.', 'data', f'hightransient_{dropout}_{trial}_{"done_allowed" if allow_done else ""}.pkl'),
                    trial_id=f'{dropout}-{trial}',
                    transient_reward=1,
                    termination_reward=0.25,
                    allow_done_action=False,
                )()
