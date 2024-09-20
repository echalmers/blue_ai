import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from constants import DATA_PATH


results = pd.read_csv(DATA_PATH / 'stress.csv', index_col=None)
results['cumulative_reward'] = results['reward'].cumsum()


plt.subplot(2,1,1)
plt.plot(results['cumulative_reward'])
plt.ylabel('cumulative reward')
plt.xlabel('step #')

plt.subplot(2,1,2)
plt.plot(results['longterm_avg'])
plt.plot(results['shortterm_avg'])
plt.legend(['long term', 'short term'])

plt.show()