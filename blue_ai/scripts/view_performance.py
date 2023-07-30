from blue_ai.scripts.train_dqn import load_trial
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


results_healthy, agent_healthy = load_trial('0.pkl')
results_dep, agent_dep = load_trial('66.pkl')
results = pd.concat((results_healthy, results_dep), ignore_index=True)

plt.figure()
sns.lineplot(data=results, x='step', y='cumulative_reward', hue='dropout')
# plt.legend(['0% dropout (healthy)', '50% dropout (depressed)'])
# plt.title('cumulative reward')

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
sns.barplot(data=goals, hue='dropout', y='count', x='goal_type')

plt.show()