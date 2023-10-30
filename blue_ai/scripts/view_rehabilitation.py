from view_performance import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


data = load_dataset('rehabilitate_*.pkl')

plt.figure()
ax = plt.gca()
x = data['step'].unique()
ax.fill_between(x, -1000, 100000, where=x < 20000, color='skyblue', alpha=0.25, linewidth=0)
ax.fill_between(x, -1000, 100000, where=(x >= 20000) & (x <= 40000), color='salmon', alpha=0.25, linewidth=0)
ax.fill_between(x, -1000, 100000, where=x > 40000, color='skyblue', alpha=0.25, linewidth=0)
sns.lineplot(data=data, x='step', y='cumulative_reward', n_boot=10, color='black')
plt.xlim((0, 60000))
plt.ylim((0, data['reward'].cumsum().max()))
plt.ylabel('cumulative reward')
plt.xlabel('time (steps in environment)')
plt.title('spine dropout modulates depression-like behaviors')

# plt.text(x=7500, y=2000, s='0% dropout', c='blue', ha='center')
# plt.text(x=20000, y=2000, s='50% dropout', c='red', ha='center')
# plt.text(x=32500, y=2000, s='0% dropout', c='blue', ha='center')

plt.show()