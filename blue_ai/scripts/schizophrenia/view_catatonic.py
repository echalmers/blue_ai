from blue_ai.scripts.train_agents import load_dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df, agents = load_dataset(filename_patterns=[
    'HealthyAgent_*.pkl',
    'SpineLossDepression_*.pkl',
    'SchizophrenicAgent_*.pkl'
], return_agents=True)

# need to add noise
for layer in row['interpretation_model'].agent.policy_net:
    if hasattr(layer, 'std'):
        print(f'changing noise std from {layer.std} to {std}')
        layer.std = std


df['pos_number'] = df['position'].apply(lambda x: np.ravel_multi_index(x, (7, 7)))
df['unique'] = df.groupby(['trial_id', 'agent'])['pos_number'].transform(lambda x: x.rolling(10).agg(lambda y: len(y.unique())))
print(df.groupby('agent')['unique'].mean())
