import pandas as pd

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.train_agents import load_trial
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from blue_ai.agents.agent_classes import HealthyAgent, SpineLossDepression


def plot_sample_envs():
    plt.figure()
    plt.subplot(1,2,1)
    env = Image2VecWrapper(TransientGoals(img_filename='env1.png', render_mode='rgb_array', transient_locations=[(2, 3), (4, 1), (5, 4)]))
    state, _ = env.reset()
    plt.imshow(env.render())
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    env = Image2VecWrapper(TransientGoals(img_filename='env1.png', render_mode='rgb_array', transient_locations=[(4, 1), (5, 4)], agent_start_pos=(2, 3), agent_start_dir=1))
    state, _ = env.reset()
    plt.imshow(env.render())
    plt.xticks([])
    plt.yticks([])


env = Image2VecWrapper(TransientGoals(img_filename='env1.png', n_transient_obstacles=0, transient_locations=[(4, 5)], agent_start_pos=(4, 4), agent_start_dir=0, render_mode='rgb_array'))
state, _ = env.reset()

def plot_perceived_value_env(ax):
    plt.sca(ax)
    ax.imshow(env.render())
    ax.set_xticks([])
    ax.set_yticks([])


def plot_perceived_value(ax):
    all_values = []

    for trial in range(20):
        for dataset in [
            f'HealthyAgent_{trial}.pkl',
            f'SpineLossDepression_{trial}.pkl',
            # f'ContextDependentLearningRate_{trial}.pkl',
            # f'HighDiscountRate_{trial}.pkl',
            # f'ScaledTargets_{trial}.pkl'
        ]:
            results, agent, _ = load_trial(os.path.join('.', 'data', dataset))

            this_agent_values = agent.get_action_values(np.expand_dims(state, 0)).numpy()
            all_values.append([agent.display_name, 'turn left', this_agent_values[0]])
            all_values.append([agent.display_name, 'turn right', this_agent_values[1]])
            all_values.append([agent.display_name, 'forward', this_agent_values[2]])

    all_values = pd.DataFrame(data=all_values, columns=['agent', 'action', 'value'])
    print(all_values)

    plt.sca(ax)
    p = sns.barplot(data=all_values, x='agent', y='value', hue='action', edgecolor=".5", hue_order=['turn left', 'forward', 'turn right'])
    p.patches[0].set_facecolor('skyblue')
    p.patches[2].set_facecolor('skyblue')
    p.patches[4].set_facecolor('skyblue')
    p.patches[1].set_facecolor('salmon')
    p.patches[3].set_facecolor('salmon')
    p.patches[5].set_facecolor('salmon')
    plt.legend([],[], frameon=False)
    plt.text(x=p.patches[0].get_center()[0], y=0.1, s='↶', fontsize='x-large', fontweight='heavy', ha='center')
    plt.text(x=p.patches[2].get_center()[0], y=0.1, s='↑', fontsize='x-large', fontweight='heavy', ha='center')
    plt.text(x=p.patches[4].get_center()[0], y=0.1, s='↷', fontsize='x-large', fontweight='heavy', ha='center')
    plt.text(x=p.patches[1].get_center()[0], y=0.1, s='↶', fontsize='x-large', fontweight='heavy', ha='center')
    plt.text(x=p.patches[3].get_center()[0], y=0.1, s='↑', fontsize='x-large', fontweight='heavy', ha='center')
    plt.text(x=p.patches[5].get_center()[0], y=0.1, s='↷', fontsize='x-large', fontweight='heavy', ha='center')

    plt.xlabel('')
    plt.ylabel('value perceived by agent')
    # plt.xticks(ticks=[0, 1], labels=['0% dropout\n(healthy)', '50% dropout\n(depressed)'])

    # fig.suptitle('anhedonia or fear generalization')


if __name__ == '__main__':
    fig, ax = plt.subplots(1, 2)

    plot_perceived_value_env(ax[0])
    plot_perceived_value(ax[1])

    plt.show()