import pandas as pd
from constants import DATA_PATH
import matplotlib.pyplot as plt



results = pd.read_csv(DATA_PATH / 'stress.csv', index_col=None)
results['cumulative_reward'] = results['reward'].cumsum()

print(results)

plt.subplot(2,1,1)
plt.plot(results['cumulative_reward'])
plt.subplot(2,1,2)
plt.plot(results['expected_short'])
plt.plot(results['expected_long'])

plt.show()