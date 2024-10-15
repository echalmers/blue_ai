import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

from blue_ai.scripts.constants import DATA_PATH


# Function to load results from a saved .pkl file
def load_results(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data['results']

# File paths (you can modify these as needed)
file_paths = DATA_PATH.glob('rehabilitate_9.pkl')


results = pd.concat([load_results(file) for file in file_paths], ignore_index=True)

# Calculate rolling average of rewards
results['rolling_avg_reward'] = results['reward'].rolling(window=8000).mean()

# Plot the rolling average
plt.plot(results['step'], results['rolling_avg_reward'])
plt.xlabel('Step')
plt.ylabel('Rolling Average Reward')
plt.title(f'Rolling Average Reward')
plt.show()