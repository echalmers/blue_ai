import pandas as pd

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.train_agents import load_trial
import blue_ai.agents.agent_classes as agent_classes
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn

from blue_ai.agents.agent_classes import HealthyAgent, SpineLossDepression


class DiscountAndCorrelationPlotter:

    def __init__(self, agent_classes=(agent_classes.HealthyAgent, agent_classes.SpineLossDepression)):

        class ActivationRecorder:

            def __init__(self):
                self.rows = [[]]

            def __call__(self, layer, intput, output):
                self.rows[-1].extend(output.numpy().flatten())

            def advance(self):
                self.rows.append([])

        all_values = []
        self.task_correlations = []

        for trial in range(20):
            for dataset in [
                f'{agent_class.__name__}_{trial}.pkl' for agent_class in agent_classes
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
                    self.task_correlations.append([trial, agent.display_name, value])

        self.all_values = pd.DataFrame(data=all_values, columns=['trial', 'agent', 'position', 'value'])
        final_values = self.all_values.groupby(['trial', 'agent']).last()['value'].reset_index().rename({'value': 'final'}, axis=1)
        self.all_values = pd.merge(self.all_values, final_values, on=['trial', 'agent'])
        self.all_values['normalized_value'] = self.all_values['value'] / self.all_values['final']
        self.all_values['estimated discount factor'] = self.all_values['normalized_value'] ** (1 / (5 - self.all_values['position']))
        self.task_correlations = pd.DataFrame(self.task_correlations, columns=['trial', 'agent', 'correlation'])

    @staticmethod
    def plot_env_locations(ax):
        env = Image2VecWrapper(
            TransientGoals(img_filename='env1.png', n_transient_goals=0, n_transient_obstacles=0,
                           agent_start_pos=(2, 6),
                           agent_start_dir=0, render_mode='rgb_array'))
        state, _ = env.reset()
        ax.imshow(env.render())
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(x=65, y=215, s='1', c='white', size='xx-large')
        ax.text(x=100, y=215, s='2', c='white', size='xx-large')
        ax.text(x=130, y=215, s='3', c='white', size='xx-large')
        ax.text(x=165, y=215, s='4', c='white', size='xx-large')

    def plot_value_of_forward(self, ax):
        plt.sca(ax)
        sns.lineplot(self.all_values, x='position', y='normalized_value', hue='agent', n_boot=10)  #, palette=['skyblue', 'salmon'])
        plt.ylabel('perceived value of moving forward\n(normalized to position 4 value)')
        plt.xticks([1, 2, 3, 4])

    def plot_inferred_discount(self, ax, **kwargs):
        plt.sca(ax)
        sns.barplot(self.all_values[self.all_values['position'] != 5], x='agent', y='estimated discount factor', n_boot=10, **kwargs)
        plt.title('inferred discount factor\n(lower means more discounting)')
        plt.ylabel('')
        plt.xlabel('')

    def plot_neuron_correlations(self, ax, **kwargs):
        plt.sca(ax)
        sns.boxplot(data=self.task_correlations, x='agent', y='correlation', **kwargs)
        # sns.violinplot(data=self.task_correlations, x='agent', y='correlation', cut=0, inner=None, **kwargs)
        plt.title('correlation of individual neuron\nactivations with position')
        plt.ylabel('')
        plt.xlabel('')


if __name__ == '__main__':
    fig, ax = plt.subplots(1, 3)
    fig2, ax2 = plt.subplots(1, 2)

    plotter = DiscountAndCorrelationPlotter()
    plotter.plot_env_locations(ax[0])
    plotter.plot_env_locations(ax2[0])

    plotter.plot_value_of_forward(ax[1])

    plotter.plot_inferred_discount(ax[2])

    plotter.plot_neuron_correlations(ax2[1])

    plt.show()