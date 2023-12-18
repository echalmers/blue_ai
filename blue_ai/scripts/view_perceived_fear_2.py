import pandas as pd

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.train_agents import load_trial
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_env_locations(ax):
    env = Image2VecWrapper(TransientGoals(img_filename='env1.png', n_transient_goals=0, transient_obstacles=[(5, 5)], agent_start_pos=(1, 5), agent_start_dir=0, render_mode='rgb_array'))
    state, _ = env.reset()
    ax.imshow(env.render())
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(x=35, y=185, s='1', c='white', size='xx-large')
    ax.text(x=70, y=185, s='2', c='white', size='xx-large')
    ax.text(x=100, y=185, s='3', c='white', size='xx-large')
    ax.text(x=135, y=185, s='4', c='white', size='xx-large')


all_values = []

for agent_pos in range(1, 5):

    env = Image2VecWrapper(TransientGoals(img_filename='env1.png', n_transient_goals=0, transient_obstacles=[(5, 5)], agent_start_pos=(agent_pos, 5), agent_start_dir=0, render_mode='rgb_array'))
    state, _ = env.reset()
    # plt.figure()
    # plt.imshow(env.render())

    for trial in range(20):
        for dataset in [
            f'HealthyAgent_{trial}.pkl',
            f'SpineLossDepression_{trial}.pkl',
            f'ContextDependentLearningRate_{trial}.pkl',
            f'HighDiscountRate_{trial}.pkl',
            f'ScaledTargets_{trial}.pkl',
            f'HighExploration_{trial}.pkl'
        ]:
            results, agent, _ = load_trial(os.path.join('.', 'data', dataset))

            this_agent_values = agent.get_action_values(np.expand_dims(state, 0)).numpy()

            all_values.append([trial, agent.display_name, agent_pos, this_agent_values[2]])


all_values = pd.DataFrame(data=all_values, columns=['trial', 'agent', 'position', 'value'])
initial_values = all_values.groupby(['trial', 'agent']).first()['value'].reset_index().rename({'value': 'initial'}, axis=1)
all_values = pd.merge(all_values, initial_values, on=['trial', 'agent'])
all_values['value'] = (all_values['value'] + 0.1) / (all_values['initial'] + 0.1)

def plot_value_of_forward(ax):
    plt.sca(ax)
    sns.lineplot(all_values, x='position', y='value', hue='agent', n_boot=10,
                 # palette=['skyblue', 'salmon']
                 )
    plt.xticks([1, 2, 3, 4])
    plt.ylabel('perceived value of moving forward\n(normalized to position 1 value)')


if __name__ == '__main__':
    fig, ax = plt.subplots(1, 2)

    plot_value_of_forward(ax[1])

    plt.show()