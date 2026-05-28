from blue_ai.scripts.train_agents import load_dataset, run_trial, load_trial
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
pd.options.display.width = 0


noise_std = 0.0

fig, axes = plt.subplots(1, 4)

# df, agents = load_dataset(filename_patterns=[
#     'HealthyAgent_*.pkl',
#     'SpineLossDepression_*.pkl',
#     'SchizophrenicAgent_*.pkl'
# ], return_agents=True)
_, agent, env = load_trial(DATA_PATH / 'HealthyAgent_0.pkl')
agents = {'file': agent}

results = []
for rep in range(1):
    for filename, agent in agents.items():
        # instantiate env
        env = Image2VecWrapper(TransientGoals(
            render_mode="human", transient_reward=0.25, termination_reward=1,
            transient_locations=[(6, 3), (4, 5)], transient_obstacles=[(4, 4)],
            agent_start_pos=(3, 4)
        ))
        state, _ = env.reset()
        # axes[0].imshow(env.render())

        # add noise
        for layer in agent.policy_net:
            if hasattr(layer, 'std'):
                print(f'changing noise std from {layer.std} to {noise_std}')
                layer.std = noise_std

        for step in range(100):
            print(step)

            # get & execute action
            action = agent.select_action(state)
            new_state, reward, done, truncated, _ = env.step(action)

            results.append({
                'filename': filename,
                'agent': agent.display_name,
                'rep': rep,
                'x': env.unwrapped.agent_pos[0],
                'y': env.unwrapped.agent_pos[1],
            })

            if truncated or done:
                break
            else:
                state = new_state


results = pd.DataFrame(results)
agents = results['agent'].unique()
max_count = 0
for i in range(len(agents)):
    this_agent_results = results[results['agent'] == agents[i]]
    hist = axes[i+1].hist2d(this_agent_results['x'], 5-this_agent_results['y'], bins=[np.arange(1, 7), np.arange(1, 7)])
    max_count = max(max_count, hist[0].max())
    print(hist[0])

print(max_count)
normalizer = Normalize(0, max_count)
im = cm.ScalarMappable(norm=normalizer)
fig.colorbar(im, ax=axes.ravel().tolist())
plt.show()