from blue_ai.scripts.train_agents import load_trial
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob
import os
from train_agents import load_dataset


def aggregate_goals(type, data, include_lava=True):
    if type == 'total':
        goals = data.groupby(by=['trial_id', 'agent'])[['terminal_goal', 'transient_goal', 'lava', 'stuck']].sum().reset_index()
    elif type == 'episode':
        goals = data.groupby(by=['trial_id', 'episode', 'agent'])[['terminal_goal', 'transient_goal', 'lava', 'stuck']].sum().reset_index()
        goals = goals.groupby(by=['trial_id', 'agent'])[['terminal_goal', 'transient_goal', 'lava', 'stuck']].mean().reset_index()

    goals_transient = goals[['trial_id', 'agent', 'transient_goal']]
    goals_transient['event'] = 'optional goal found'
    goals_transient.rename({'transient_goal': 'count'}, axis=1, inplace=True)

    goals_terminal = goals[['trial_id', 'agent', 'terminal_goal']]
    goals_terminal['event'] = 'required goal found'
    goals_terminal.rename({'terminal_goal': 'count'}, axis=1, inplace=True)

    lava = goals[['trial_id', 'agent', 'lava']]
    lava['event'] = 'lava found'
    lava.rename({'lava': 'count'}, axis=1, inplace=True)

    # stuck = goals[['trial_id', 'dropout', 'stuck']]
    # stuck['goal type'] = 'stuck'
    # stuck.rename({'stuck': 'count'}, axis=1, inplace=True)

    goals = pd.concat(
        [goals_terminal, goals_transient] + ([lava] if include_lava else [])
        , ignore_index=True)
    return goals


if __name__ == '__main__':
    n_boot = 1

    high_terminal_results = load_dataset(['HealthyAgent_?.pkl', 'SpineLossDepression_?.pkl', 'ContextDependentLearningRate_?.pkl'])

    # plot cumulative reward
    fig = plt.figure()
    fig.suptitle('depressed agent shows simpler, less-rewarding behavior')

    plt.subplot(1,2,1)
    sns.lineplot(data=high_terminal_results, x='step', y='cumulative_reward', hue='agent', n_boot=n_boot, palette=['skyblue', 'salmon', 'red'])
    plt.ylabel('cumulative reward obtained')
    plt.xlabel('time (steps in environment)')
    plt.xticks([0, 30000])

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
    plt.subplot(1,2,2)
    high_terminal_goals = aggregate_goals(type='episode', data=high_terminal_results)
    sns.barplot(data=high_terminal_goals, x='agent', y='count', hue='event', n_boot=n_boot,
                palette=['tab:green', 'tab:blue', 'tab:red'])
    plt.ylabel('goals obtained per episode')
    # sns.move_legend(plt.gca(), "upper left")
    plt.xlabel('type of goal')
    plt.xlabel('')
    # plt.xticks(ticks=[0, 1], labels=['0% dropout\n(healthy)', '50% dropout\n(depressed)'])

    plt.show()