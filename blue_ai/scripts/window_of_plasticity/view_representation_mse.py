import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from blue_ai.scripts.constants import DATA_PATH
pd.options.display.width = 0


df = pd.read_csv(DATA_PATH / 'rehabilitation_tests_1_4.csv', index_col=None)
df = df.dropna(subset='reconstruction_mse')
df['reconstruction_mse'] = df.groupby(['agent', 'trial_id'])['reconstruction_mse'].transform(lambda x: x.rolling(10).mean())

# reduce dataset
df = df[(df['agent'] == 'SpineLossDepression') & (df['step'] < 41_000) & (df['step'] > 39_000)]
sns.lineplot(data=df, x='step', y='reconstruction_mse', hue='path', errorbar='sd')
plt.grid()
plt.show()

