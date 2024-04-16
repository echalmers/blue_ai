import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

results = pd.read_csv(os.path.join('.', 'data', 'simple_env_comparison.csv'))
results['reward'] *= 100
print(results)
sns.catplot(data=results, kind='bar', x='env', y='reward', hue='agent')
plt.xlabel('environment')
plt.ylabel('average reward per 100 steps')
plt.show()