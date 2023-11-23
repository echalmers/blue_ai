from view_performance import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


data = load_dataset('rehabilitate_*.pkl')

avg_results = data.groupby('step')['cumulative_reward'].mean().reset_index()  # avg_results is a dataframe that has the average cumulative reward curve

plt.figure()
ax = plt.gca()
x = data['step'].unique()
ax.fill_between(x, -1000, 100000, where=x < 20000, color='skyblue', alpha=0.25, linewidth=0)
ax.fill_between(x, -1000, 100000, where=(x >= 20000) & (x <= 40000), color='salmon', alpha=0.25, linewidth=0)
ax.fill_between(x, -1000, 100000, where=x > 40000, color='skyblue', alpha=0.25, linewidth=0)
sns.lineplot(data=data, x='step', y='cumulative_reward', n_boot=1, color='black')
# plt.xlim((0, 60000))
plt.ylim((0, data['cumulative_reward'].max()))
plt.ylabel('cumulative reward')
plt.xlabel('time (steps in environment)')
plt.title('spine dropout modulates depression-like behaviors')

plt.text(x=10_000, y=2000, s='healthy', c='blue', ha='center')
plt.text(x=30_000, y=2000, s='simulated spine loss', c='red', ha='center')
plt.text(x=50_000, y=2000, s='spines restored', c='blue', ha='center')

plt.show()