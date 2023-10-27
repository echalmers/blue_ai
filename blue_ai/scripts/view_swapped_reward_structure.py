import matplotlib.pyplot as plt
import seaborn as sns
from view_performance import load_dataset, aggregate_goals
from blue_ai.agents.agent_classes import HealthyAgent, SpineLossDepression


n_boot = 1

high_terminal_results = load_dataset(['HealthyAgent_?.pkl', 'SpineLossDepression_?.pkl'])
high_transient_results = load_dataset(['HealthyAgent_swapped_?.pkl', 'SpineLossDepression_swapped_?.pkl'])

plt.figure()
p = plt.subplot(1,2,1)
high_terminal_goals = aggregate_goals(type='episode', data=high_terminal_results, include_lava=False)
sns.barplot(data=high_terminal_goals, x='agent', y='count', hue='event', n_boot=n_boot, palette=['tab:green', 'tab:blue'])
plt.ylabel('goals obtained per episode')
plt.title('normal reward scheme')
# sns.move_legend(p, "upper left")
plt.xlabel('type of goal')
plt.xlabel('')
# plt.xticks(ticks=[0, 1], labels=['0% dropout\n(healthy)', '50% dropout\n(depressed)'])

ax = plt.subplot(1,2,2, sharey=p)
high_transient_goals = aggregate_goals(type='episode', data=high_transient_results, include_lava=False)
sns.barplot(data=high_transient_goals, x='agent', y='count', hue='event', n_boot=n_boot, palette=['tab:green', 'tab:blue'])
plt.title('high reward for optional goals')
plt.ylabel('')
ax.yaxis.tick_right()
plt.legend([],[], frameon=False)
plt.xlabel('')
# plt.xticks(ticks=[0, 1], labels=['0% dropout\n(healthy)', '50% dropout\n(depressed)'])

# add annotations showing increase
transient_count_1 = high_terminal_goals[high_terminal_goals['event'] == 'optional goal found'][['agent', 'count']].set_index('agent')
transient_count_2 = high_transient_goals[high_transient_goals['event'] == 'optional goal found'][['agent', 'count']].set_index('agent')
transient_count_2 = transient_count_2.groupby('agent').mean()
transient_count_1 = transient_count_1.groupby('agent').mean()
plt.arrow(x=0.6, y=transient_count_1.loc[HealthyAgent.display_name, 'count'], dx=0, dy=transient_count_2.loc[HealthyAgent.display_name, 'count']-transient_count_1.loc[HealthyAgent.display_name, 'count'], head_width=0.1, length_includes_head=True, overhang=1)
plt.plot([0.2, 0.6], [transient_count_1.loc[HealthyAgent.display_name, 'count']]*2, c='k')
plt.arrow(x=1.6, y=transient_count_1.loc[SpineLossDepression.display_name, 'count'], dx=0, dy=transient_count_2.loc[SpineLossDepression.display_name, 'count']-transient_count_1.loc[SpineLossDepression.display_name, 'count'], head_width=0.1, length_includes_head=True, overhang=1)
plt.plot([1.2, 1.6], [transient_count_1.loc[SpineLossDepression.display_name, 'count']]*2, c='k')
plt.annotate(f"+{round((transient_count_2.loc[HealthyAgent.display_name, 'count']-transient_count_1.loc[HealthyAgent.display_name, 'count'])/transient_count_1.loc[HealthyAgent.display_name, 'count']*100)}%",
             xy=(0.4, transient_count_2.loc[HealthyAgent.display_name, 'count']+0.05))
plt.annotate(f"+{round((transient_count_2.loc[SpineLossDepression.display_name, 'count']-transient_count_1.loc[SpineLossDepression.display_name, 'count'])/transient_count_1.loc[SpineLossDepression.display_name, 'count']*100)}%",
             xy=(1.4, transient_count_2.loc[SpineLossDepression.display_name, 'count']+0.05))

xlim = plt.gca().get_xlim()
plt.xlim((xlim[0], 1.9))
ylim = plt.gca().get_ylim()
plt.ylim((ylim[0], ylim[1] + 0.05))

plt.show()