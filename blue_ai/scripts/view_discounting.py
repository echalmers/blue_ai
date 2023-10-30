import pandas as pd

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.train_agents import load_trial
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from blue_ai.agents.agent_classes import HealthyAgent, SpineLossDepression





fig, ax = plt.subplots(1, 3)

env = Image2VecWrapper(TransientGoals(img_filename='env1.png', n_transient_goals=0, n_transient_obstacles=0, agent_start_pos=(2, 6), agent_start_dir=0, render_mode='rgb_array'))
state, _ = env.reset()
ax[0].imshow(env.render())
ax[0].set_xticks([])
ax[0].set_yticks([])

all_values = []
initial_final_values = pd.DataFrame()

for agent_pos in range(2, 6):

    env = Image2VecWrapper(TransientGoals(img_filename='env1.png', n_transient_goals=0, n_transient_obstacles=0, agent_start_pos=(agent_pos, 6), agent_start_dir=0, render_mode='rgb_array'))
    state, _ = env.reset()
    # plt.figure()
    # plt.imshow(env.render())

    for trial in range(10):
        for dataset in [f'HealthyAgent_{trial}.pkl', f'SpineLossDepression_{trial}.pkl', f'ContextDependentLearningRate_{trial}.pkl']:
            results, agent, _ = load_trial(os.path.join('.', 'data', dataset))

            if agent_pos == 2:
                initial_final_values = pd.concat((initial_final_values, pd.DataFrame([{'trial': trial, 'agent': agent.display_name}])), ignore_index=True)

            this_agent_value = agent.get_action_values(np.expand_dims(state, 0)).numpy().max()
            all_values.append([agent.display_name, agent_pos, this_agent_value])

            if agent_pos in (2, 5):
                initial_final_values.loc[(initial_final_values['trial'] == trial) & (initial_final_values['agent'] == agent.display_name), agent_pos] = this_agent_value


all_values = pd.DataFrame(data=all_values, columns=['agent', 'position', 'value'])
initial_final_values['estimated discount factor'] = (initial_final_values[2] / initial_final_values[5]) ** 0.25

plt.sca(ax[1])
sns.lineplot(all_values, x='position', y='value', hue='agent', n_boot=10)

plt.sca(ax[2])
sns.barplot(initial_final_values, x='agent', y='estimated discount factor', n_boot=10)

plt.show()