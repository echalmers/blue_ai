import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from blue_ai.scripts.constants import DATA_PATH
pd.options.display.width = 0


df = pd.read_csv(DATA_PATH / 'rehabilitation_tests.csv', index_col=None)

# normalize agent performance to healthy
df['reward_normalized'] = df['reward'] / df['reward_healthy']

# rolling avg
# df['reward_avg_normalized'] = df['reward_normalized'].rolling(1000).mean()
# df['reward_avg'] = df['reward'].rolling(1000).mean()
df['reward_avg'] = df.groupby(['agent', 'trial_id'])['reward'].transform(lambda x: x.rolling(2000).mean())
df['reward_avg_normalized'] = df.groupby(['agent', 'trial_id'])['reward_normalized'].transform(lambda x: x.rolling(2000).mean())

# reduce resolution for faster plotting
df.drop(df[df['step'] % 20 != 0].index, inplace=True)


ax1 = plt.subplot(1, 3, 1)
sns.lineplot(data=df[df['agent'] == 'SpineLossDepression'], x='step', y='reward_avg', hue='path', errorbar='sd')

plt.subplot(1, 3, 2, sharex=ax1)
sns.lineplot(data=df[df['agent'] == 'SpineLossDepression'], x='step', y='reward_avg_normalized', hue='path', errorbar='sd')

plt.subplot(1, 3, 3, sharex=ax1)
sns.lineplot(data=df[df['agent'] == 'SpineLossDepression'], x='step', y='cumulative_reward', hue='path', errorbar='sd')

plt.grid()



plt.show()
