import pandas as pd

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.train_agents import load_trial
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn

from blue_ai.agents.agent_classes import HealthyAgent, SpineLossDepression





fig, ax = plt.subplots(1, 2)

env = Image2VecWrapper(TransientGoals(img_filename='env1.png', n_transient_goals=0, transient_obstacles=[(5, 5)], agent_start_pos=(1, 5), agent_start_dir=0, render_mode='rgb_array'))
state, _ = env.reset()
ax[0].imshow(env.render())
ax[0].set_xticks([])
ax[0].set_yticks([])

all_values = []

for agent_pos in range(1, 5):

    env = Image2VecWrapper(TransientGoals(img_filename='env1.png', n_transient_goals=0, transient_obstacles=[(5, 5)], agent_start_pos=(agent_pos, 5), agent_start_dir=0, render_mode='rgb_array'))
    state, _ = env.reset()
    # plt.figure()
    # plt.imshow(env.render())

    for trial in range(10):
        for dataset in [f'HealthyAgent_{trial}.pkl', f'SpineLossDepression_{trial}.pkl', f'ContextDependentLearningRate_{trial}.pkl']:
            results, agent, _ = load_trial(os.path.join('.', 'data', dataset))

            this_agent_values = agent.get_action_values(np.expand_dims(state, 0)).numpy()
            preferred_dir = np.argmax(this_agent_values)

            all_values.append([agent.display_name, agent_pos, preferred_dir])


all_values = pd.DataFrame(data=all_values, columns=['agent', 'position', 'value'])

plt.sca(ax[1])
sns.lineplot(all_values, x='position', y='value', hue='agent')

plt.show()