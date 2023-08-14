from blue_ai.scripts.train_dqn import load_trial, Image2VecWrapper, TransientGoals
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob
import os


def load_dataset(filename_pattern):
    results = []
    for filename in glob.glob(os.path.join('.', 'data', filename_pattern)):
        print(filename)
        this_result, _, _ = load_trial(filename)
        results.append(this_result)
    results = pd.concat(results, ignore_index=True)
    results['dropout'].replace({0: '0% (healthy)', 50: '50% (depressed)'}, inplace=True)
    return results


def aggregate_goals(type, data):
    if type == 'total':
        goals = data.groupby(by=['trial_id', 'dropout'])[['terminal_goal', 'transient_goal']].sum().reset_index()
    elif type == 'episode':
        goals = data.groupby(by=['trial_id', 'episode', 'dropout'])[['terminal_goal', 'transient_goal']].sum().reset_index()
        goals = goals.groupby(by=['trial_id', 'dropout'])[['terminal_goal', 'transient_goal']].mean().reset_index()

    goals_transient = goals[['trial_id', 'dropout', 'transient_goal']]
    goals_transient['goal type'] = 'optional'
    goals_transient.rename({'transient_goal': 'count'}, axis=1, inplace=True)
    goals_terminal = goals[['trial_id', 'dropout', 'terminal_goal']]
    goals_terminal['goal type'] = 'required'
    goals_terminal.rename({'terminal_goal': 'count'}, axis=1, inplace=True)
    goals = pd.concat((goals_terminal, goals_transient), ignore_index=True)
    goals['goal type'].replace({'terminal': 'required'}, inplace=True)
    return goals


if __name__ == '__main__':
    high_terminal_results = load_dataset('highterminal*_.pkl')
    high_transient_results = load_dataset('hightransient*_.pkl')

    # plot cumulative reward
    fig = plt.figure()
    fig.suptitle('depressed agent shows simpler, less-rewarding behavior')

    plt.subplot(1,2,1)
    sns.lineplot(data=high_terminal_results, x='step', y='cumulative_reward', hue='dropout', n_boot=100, palette=['skyblue', 'salmon'])
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
    sns.barplot(data=high_terminal_goals, x='dropout', y='count', hue='goal type', n_boot=100,
                palette=['tab:green', 'tab:blue'])
    plt.ylabel('goals obtained per episode')
    # sns.move_legend(plt.gca(), "upper left")
    plt.xlabel('type of goal')
    plt.xlabel('')
    plt.xticks(ticks=[0, 1], labels=['0% dropout\n(healthy)', '50% dropout\n(depressed)'])

    #goals reached per episode compared with high transient goals
    plt.figure()
    p = plt.subplot(1,2,1)
    high_terminal_goals = aggregate_goals(type='episode', data=high_terminal_results)
    sns.barplot(data=high_terminal_goals, x='dropout', y='count', hue='goal type', n_boot=100, palette=['tab:green', 'tab:blue'])
    plt.ylabel('goals obtained per episode')
    plt.title('normal reward scheme')
    # sns.move_legend(p, "upper left")
    plt.xlabel('type of goal')
    plt.xlabel('')
    plt.xticks(ticks=[0, 1], labels=['0% dropout\n(healthy)', '50% dropout\n(depressed)'])

    ax = plt.subplot(1,2,2, sharey=p)
    high_transient_goals = aggregate_goals(type='episode', data=high_transient_results)
    sns.barplot(data=high_transient_goals, x='dropout', y='count', hue='goal type', n_boot=100, palette=['tab:green', 'tab:blue'])
    plt.title('high reward for optional goals')
    plt.ylabel('')
    ax.yaxis.tick_right()
    plt.legend([],[], frameon=False)
    plt.xlabel('')
    plt.xticks(ticks=[0, 1], labels=['0% dropout\n(healthy)', '50% dropout\n(depressed)'])

    # add annotations showing increase
    transient_count_1 = high_terminal_goals[high_terminal_goals['goal type'] == 'optional'][['dropout', 'count']].set_index('dropout')
    transient_count_2 = high_transient_goals[high_transient_goals['goal type'] == 'optional'][['dropout', 'count']].set_index('dropout')
    transient_count_2 = transient_count_2.groupby('dropout').mean()
    transient_count_1 = transient_count_1.groupby('dropout').mean()
    plt.arrow(x=0.6, y=transient_count_1.loc['0% (healthy)', 'count'], dx=0, dy=transient_count_2.loc['0% (healthy)', 'count']-transient_count_1.loc['0% (healthy)', 'count'], head_width=0.1, length_includes_head=True, overhang=1)
    plt.plot([0.2, 0.6], [transient_count_1.loc['0% (healthy)', 'count']]*2, c='k')
    plt.arrow(x=1.6, y=transient_count_1.loc['50% (depressed)', 'count'], dx=0, dy=transient_count_2.loc['50% (depressed)', 'count']-transient_count_1.loc['50% (depressed)', 'count'], head_width=0.1, length_includes_head=True, overhang=1)
    plt.plot([1.2, 1.6], [transient_count_1.loc['50% (depressed)', 'count']]*2, c='k')
    plt.annotate(f"+{round((transient_count_2.loc['0% (healthy)', 'count']-transient_count_1.loc['0% (healthy)', 'count'])/transient_count_1.loc['0% (healthy)', 'count']*100)}%",
                 xy=(0.4, transient_count_2.loc['0% (healthy)', 'count']+0.05))
    plt.annotate(f"+{round((transient_count_2.loc['50% (depressed)', 'count']-transient_count_1.loc['50% (depressed)', 'count'])/transient_count_1.loc['50% (depressed)', 'count']*100)}%",
                 xy=(1.4, transient_count_2.loc['50% (depressed)', 'count']+0.05))

    xlim = plt.gca().get_xlim()
    plt.xlim((xlim[0], 1.9))
    ylim = plt.gca().get_ylim()
    plt.ylim((ylim[0], ylim[1] + 0.05))

    plt.show()