import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

results = pd.read_csv(os.path.join('.', 'data', 'decay_sweep.csv'))
results['reward'] *= 100
print(results)
plt.figure(figsize=(8, 4))
sns.lineplot(results, x='decay', y='reward', palette=['grey'])
plt.xscale('log')
plt.xlabel('weight decay (λ)')
plt.ylabel('average reward per 100 steps')
plt.title('performance as weight decay increases')
plt.tight_layout()
plt.savefig('img/decay_sweep.png', dpi=300)
plt.show()