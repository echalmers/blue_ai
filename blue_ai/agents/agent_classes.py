import copy

from blue_ai.agents.dqn import DQN
from torch import nn
from copy import deepcopy


# a basic multi-layer network
common_network = nn.Sequential(
    nn.Flatten(1, -1),
    nn.Linear(100, 10),
    nn.Tanh(),
    nn.Linear(10, 3)
)


class BaseAgent(DQN):

    def __init__(self,
                 input_shape=(4, 5, 5),
                 replay_buffer_size=10000,
                 update_frequency=5,
                 lr=0.01,
                 sync_frequency=25,
                 gamma=0.9,
                 epsilon=0.05,
                 batch_size=1500,
                 weight_decay=0.0
                 ):
        super().__init__(
            network=nn.Sequential(
                nn.Flatten(1, -1),
                nn.Linear(100, 10),
                nn.Tanh(),
                nn.Linear(10, 3)
            ),
            input_shape=input_shape,
            replay_buffer_size=replay_buffer_size,
            update_frequency=update_frequency,
            lr=lr,
            sync_frequency=sync_frequency,
            gamma=gamma,
            epsilon=epsilon,
            batch_size=batch_size,
            weight_decay=weight_decay
        )


class HealthyAgent(BaseAgent):

    display_name = 'control'

    def __init__(self):
        super().__init__(weight_decay=0)


class SpineLossDepression(BaseAgent):

    display_name = 'simulated spine loss'

    def __init__(self):
        super().__init__(weight_decay=2.5e-3)


class ContextDependentLearningRate(BaseAgent):

    display_name = 'context-dependent learning rate'
    positive_scale = 0.5
    negative_scale = 2

    def update(self, state, action, reward, new_state, done):
        if reward > 0:
            reward *= self.positive_scale
        elif reward < 0:
            reward *= self.negative_scale

        super().update(state, action, reward, new_state, done)