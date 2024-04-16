import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

results = pd.read_csv(os.path.join('.', 'data', 'decay_sweep.csv'))
results['reward'] *= 100
print(results)
sns.lineplot(results, x='decay', y='reward')
plt.xscale('log')
plt.xlabel('weight decay (λ)')
plt.ylabel('average reward per 100 steps')
plt.show()