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


class ActivationRecorder:

    def __init__(self):
        self.rows = [[]]

    def __call__(self, layer, intput, output):
        self.rows[-1].extend(output.numpy().flatten())
    def advance(self):
        self.rows.append([])


fig, ax = plt.subplots(1, 3)
fig2, ax2 = plt.subplots(1, 2)

env = Image2VecWrapper(TransientGoals(img_filename='env1.png', n_transient_goals=0, n_transient_obstacles=0, agent_start_pos=(2, 6), agent_start_dir=0, render_mode='rgb_array'))
state, _ = env.reset()
for a in [ax, ax2]:
    a[0].imshow(env.render())
    a[0].set_xticks([])
    a[0].set_yticks([])
    a[0].text(x=65, y=215, s='1', c='white', size='xx-large')
    a[0].text(x=100, y=215, s='2', c='white', size='xx-large')
    a[0].text(x=130, y=215, s='3', c='white', size='xx-large')
    a[0].text(x=165, y=215, s='4', c='white', size='xx-large')

all_values = []
task_correlations = []

for trial in range(20):
    for dataset in [
        # f'HealthyAgent_{trial}.pkl',
        f'SpineLossDepression_{trial}.pkl',
        # f'ContextDependentLearningRate_{trial}.pkl',
        # f'HighDiscountRate_{trial}.pkl',
        # f'ShiftedTargets_{trial}.pkl',
    ]:
        results, agent, _ = load_trial(os.path.join('.', 'data', dataset))
        recorder = ActivationRecorder()
        for layer in agent.policy_net:
            if isinstance(layer, nn.Linear):
                layer.register_forward_hook(recorder)

        for agent_pos in range(2, 6):

            state, _ = Image2VecWrapper(TransientGoals(img_filename='env1.png', n_transient_goals=0, n_transient_obstacles=0,
                                                    agent_start_pos=(agent_pos, 6), agent_start_dir=0,
                                                    render_mode='rgb_array')).reset()

            this_agent_value = agent.get_action_values(np.expand_dims(state, 0)).numpy().max()
            all_values.append([trial, agent.display_name, agent_pos, this_agent_value])
            recorder.advance()

        corr_df = pd.DataFrame(recorder.rows[:-1])
        corr_df['pos'] = range(4)
        for value in corr_df.corr()['pos'].abs().drop('pos'):
            task_correlations.append([trial, agent.display_name, value])

all_values = pd.DataFrame(data=all_values, columns=['trial', 'agent', 'position', 'value'])
final_values = all_values.groupby(['trial', 'agent']).last()['value'].reset_index().rename({'value': 'final'}, axis=1)
all_values = pd.merge(all_values, final_values, on=['trial', 'agent'])
all_values['normalized_value'] = all_values['value'] / all_values['final']
all_values['estimated discount factor'] = all_values['normalized_value'] ** (1 / (5 - all_values['position']))

plt.sca(ax[1])
sns.lineplot(all_values, x='position', y='normalized_value', hue='agent', n_boot=10)  #, palette=['skyblue', 'salmon'])
plt.ylabel('perceived value of moving forward\n(normalized to position 4 value)')
plt.xticks([1, 2, 3, 4])

plt.sca(ax[2])
sns.barplot(all_values[all_values['position'] != 5], x='agent', y='estimated discount factor', n_boot=10)  #, palette=['skyblue', 'salmon'])

plt.sca(ax2[1])
task_correlations = pd.DataFrame(data=task_correlations, columns=['trial', 'agent', 'correlation'])
sns.boxplot(data=task_correlations, x='agent', y='correlation')
plt.ylabel('correlation of individual neuron activation\nwith position')


plt.show()