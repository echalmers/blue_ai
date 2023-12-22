from blue_ai.scripts.train_agents import load_trial
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob
import os
from train_agents import load_dataset
from blue_ai.envs.transient_goals import TransientGoals
import blue_ai.agents.agent_classes as agent_classes



class PerformancePlotter:

    def __init__(self, agent_classes=(agent_classes.HealthyAgent, agent_classes.SpineLossDepression)):

        self.high_terminal_results = load_dataset([
                f'{cls.__name__}_[!s]*.pkl' for cls in agent_classes
            ])

    @staticmethod
    def plot_sample_env(ax):
        plt.sca(ax)

        # plot sample env
        env = TransientGoals(img_filename='env1.png', transient_locations=[(4, 1), (5, 4)],
                             transient_obstacles=[(2, 5)],
                             agent_start_pos=(2, 3), agent_start_dir=1, render_mode='rgb_array')
        env.reset()

        plt.imshow(env.render())
        plt.xticks([])
        plt.yticks([])

    @staticmethod
    def aggregate_goals(type, data, include_lava=True):
        if type == 'total':
            goals = data.groupby(by=['trial_id', 'agent'])[
                ['terminal_goal', 'transient_goal', 'lava', 'stuck']].sum().reset_index()
        elif type == 'episode':
            goals = data.groupby(by=['trial_id', 'episode', 'agent'])[
                ['terminal_goal', 'transient_goal', 'lava', 'stuck']].sum().reset_index()
            goals = goals.groupby(by=['trial_id', 'agent'])[
                ['terminal_goal', 'transient_goal', 'lava', 'stuck']].mean().reset_index()

        goals_transient = goals[['trial_id', 'agent', 'transient_goal']]
        goals_transient['event'] = 'optional goal'
        goals_transient.rename({'transient_goal': 'count'}, axis=1, inplace=True)

        goals_terminal = goals[['trial_id', 'agent', 'terminal_goal']]
        goals_terminal['event'] = 'required goal'
        goals_terminal.rename({'terminal_goal': 'count'}, axis=1, inplace=True)

        lava = goals[['trial_id', 'agent', 'lava']]
        lava['event'] = 'hazard'
        lava.rename({'lava': 'count'}, axis=1, inplace=True)

        # stuck = goals[['trial_id', 'dropout', 'stuck']]
        # stuck['goal type'] = 'stuck'
        # stuck.rename({'stuck': 'count'}, axis=1, inplace=True)

        goals = pd.concat(
            [goals_terminal, goals_transient] + ([lava] if include_lava else [])
            , ignore_index=True)
        goals.rename({'event': 'object'}, axis=1, inplace=True)
        return goals

    def plot_learning_curves(self, ax, n_boot=1, **kwargs):
        plt.sca(ax)

        # high_terminal_results['avg_reward'] = high_terminal_results.groupby(['trial_id', 'agent'])['reward'].transform(lambda x: x.rolling(250).mean())

        # plot cumulative reward
        sns.lineplot(data=self.high_terminal_results[(self.high_terminal_results['step'] <= 20_000) & (self.high_terminal_results['step'] % 5 == 0)],
                     x='step', y='cumulative_reward',
                     hue='agent', n_boot=n_boot,
                     **kwargs
                     )
        plt.title('cumulative reward obtained')
        plt.ylabel('')
        plt.xlabel('time (steps in environment)')

    def plot_goals_per_episode(self, ax, n_boot=1):
        plt.sca(ax)

        high_terminal_goals = self.aggregate_goals(type='episode', data=self.high_terminal_results)
        sns.barplot(data=high_terminal_goals, x='agent', y='count', hue='object', n_boot=n_boot,
                    palette=['tab:green', 'tab:blue', 'tab:red'])
        plt.title('objects reached per episode')
        plt.ylabel('')
        # sns.move_legend(plt.gca(), "upper left")
        plt.xlabel('type of goal')
        plt.xlabel('')


if __name__ == '__main__':

    plotter = PerformancePlotter()

    f, ax = plt.subplots(1, 2, figsize=(9, 3))
    # plot_sample_env(ax[0])

    plt.subplot(1, 2, 1)
    plotter.plot_learning_curves(ax[0])

    plt.subplot(1, 2, 2)
    plotter.plot_goals_per_episode(ax[1])

    plt.show()
    exit()




    # plot cumulative reward

    plt.subplot(1,3,2)
    sns.lineplot(data=high_terminal_results[high_terminal_results['step'] <= 20_000], x='step', y='cumulative_reward', hue='agent', n_boot=n_boot, palette=['skyblue', 'salmon', 'red'])
    plt.ylabel('cumulative reward obtained')
    plt.xlabel('time (steps in environment)')
    plt.xticks([0, 20000])
    plt.text(x=10_000, y=500, s='healthy', c='skyblue')
    plt.text(x=10_000, y=100, s='simulated spine loss', c='salmon')
    plt.legend([], [], frameon=False)

    # # steps per goal
    # plt.figure()
    # steps = results.groupby(by=['trial_id', 'dropout', 'episode'])['step'].count().reset_index()
    # steps.rename({'step': 'steps'}, axis=1, inplace=True)
    # sns.lineplot(data=steps, x='episode', y='steps', hue='dropout', n_boot=100)

    #total goals reached
    # plt.figure()
    # p = plt.subplot(1,2,1)
    # high_terminal_goals = aggregate_goals(type='total', data=high_terminal_results)
    # sns.barplot(data=high_terminal_goals, x='dropout', y='count', hue='goal_type', n_boot=100, palette=['tab:blue', 'tab:green'])
    # plt.ylabel('total rewards obtained')
    # plt.title('normal')
    #
    # plt.subplot(1,2,2, sharey=p)
    # high_transient_goals = aggregate_goals(type='total', data=high_transient_results)
    # sns.barplot(data=high_transient_goals, x='dropout', y='count', hue='goal_type', n_boot=100, palette=['tab:blue', 'tab:green'])
    # plt.title('high transient rewards')

    # goals reached per episode
    plt.subplot(1,3,3)
    high_terminal_goals = aggregate_goals(type='episode', data=high_terminal_results)
    sns.barplot(data=high_terminal_goals, x='agent', y='count', hue='event', n_boot=n_boot,
                palette=['tab:green', 'tab:blue', 'tab:red'])
    plt.ylabel('goals obtained per episode')
    # sns.move_legend(plt.gca(), "upper left")
    plt.xlabel('type of goal')
    plt.xlabel('')
    # plt.xticks(ticks=[0, 1], labels=['0% dropout\n(healthy)', '50% dropout\n(depressed)'])

    plt.show()