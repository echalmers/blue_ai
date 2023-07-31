from blue_ai.scripts.train_dqn import load_trial
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob
import os


results = []
for filename in glob.glob(os.path.join('.', 'data', 'hightransient*.pkl')):
    print(filename)
    this_result, _ = load_trial(filename)
    results.append(this_result)
results = pd.concat(results, ignore_index=True)


# cumulative reward
plt.figure()
sns.lineplot(data=results, x='step', y='cumulative_reward', hue='dropout', n_boot=100)
# plt.legend(['0% dropout (healthy)', '50% dropout (depressed)'])
# plt.title('cumulative reward')

# steps per goal
plt.figure()
steps = results.groupby(by=['trial_id', 'dropout', 'episode'])['step'].count().reset_index()
steps.rename({'step': 'steps'}, axis=1, inplace=True)
sns.lineplot(data=steps, x='episode', y='steps', hue='dropout', n_boot=100)

# total goals reached
plt.figure()
goals = results.groupby(by=['trial_id', 'dropout'])[['terminal_goal', 'transient_goal']].sum().reset_index()
goals_transient = goals[['trial_id', 'dropout', 'transient_goal']]
goals_transient['goal_type'] = 'transient'
goals_transient.rename({'transient_goal': 'count'}, axis=1, inplace=True)
goals_terminal = goals[['trial_id', 'dropout', 'terminal_goal']]
goals_terminal['goal_type'] = 'terminal'
goals_terminal.rename({'terminal_goal': 'count'}, axis=1, inplace=True)
goals = pd.concat((goals_terminal, goals_transient), ignore_index=True)
print(goals)
sns.barplot(data=goals, hue='dropout', y='count', x='goal_type', n_boot=100)
plt.ylabel('total rewards obtained')

# goals reached per episode
plt.figure()
goals = results.groupby(by=['trial_id', 'episode', 'dropout'])[['terminal_goal', 'transient_goal']].sum().reset_index()
goals = goals.groupby(by=['trial_id', 'dropout'])[['terminal_goal', 'transient_goal']].mean().reset_index()
goals_transient = goals[['trial_id', 'dropout', 'transient_goal']]
goals_transient['goal_type'] = 'transient'
goals_transient.rename({'transient_goal': 'count'}, axis=1, inplace=True)
goals_terminal = goals[['trial_id', 'dropout', 'terminal_goal']]
goals_terminal['goal_type'] = 'terminal'
goals_terminal.rename({'terminal_goal': 'count'}, axis=1, inplace=True)
goals = pd.concat((goals_terminal, goals_transient), ignore_index=True)
print(goals)
sns.barplot(data=goals, hue='dropout', y='count', x='goal_type', n_boot=100)
plt.ylabel('rewards obtained per episode')

plt.show()