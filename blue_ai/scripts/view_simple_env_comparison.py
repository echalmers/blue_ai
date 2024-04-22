import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

results = pd.read_csv(os.path.join('.', 'data', 'simple_env_comparison.csv'))
results['agent'] = results['agent'].replace({'HealthyAgent': 'healthy', 'SpineLossDepression': 'simluated\nspine loss'})
results['reward'] *= 100
print(results)
p = sns.catplot(data=results, kind='bar', x='env', y='reward', hue='agent', palette=['skyblue', 'salmon'], legend_out=False)
p.fig.set_size_inches(9, 4)
# sns.move_legend(p, 'upper left')
plt.xlabel('environment')
plt.ylabel('average reward per 100 steps')
plt.title('performance in full-complexity and simple environments')
plt.tight_layout()
plt.savefig('img/simple_env_comparison.png', dpi=300)
plt.show()