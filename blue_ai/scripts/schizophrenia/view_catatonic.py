from blue_ai.scripts.train_agents import load_dataset, run_trial
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.options.display.width = 0


noise_std = 0.4
window_size = 20

env = Image2VecWrapper(TransientGoals(render_mode="rgb_array", transient_reward=0.25, termination_reward=1))

df, agents = load_dataset(filename_patterns=[
    'HealthyAgent_*.pkl',
    'SpineLossDepression_*.pkl',
    'SchizophrenicAgent_*.pkl'
], return_agents=True)

results = []
for filename, agent in agents.items():
    # add noise
    for layer in agent.policy_net:
        if hasattr(layer, 'std'):
            print(f'changing noise std from {layer.std} to {noise_std}')
            layer.std = noise_std

    df, _, _ = run_trial(agent=agent, env=env, trial_id=list(agents).index(filename), steps=5000)
    df = df.tail(3000)
    df['pos_number'] = df['position'].apply(lambda x: np.ravel_multi_index(x, (7, 7)))
    df['unique'] = df['pos_number'].rolling(window_size).agg(lambda y: len(y.unique()))
    results.append({
        'filename': filename,
        'agent': agent.display_name,
        'repeats': (window_size - df['unique']).mean(),
        'std': (window_size - df['unique']).std()
    })

results = pd.DataFrame(results)
print(results)
