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
import matplotlib.transforms as mtransforms
pd.options.display.width = 0

# set up plot window
mosaic = """
abb
...
cde
"""
fig, axes = plt.subplot_mosaic(mosaic, figsize=(9, 4), height_ratios=[0.45, 0.1, 0.45])
noise_levels = {
    0: 'c',
    0.1: None,
    0.2: 'd',
    0.3: None,
    0.4: 'e',
    # 0.5: None,
}

# instantiate env
env = Image2VecWrapper(TransientGoals(
    render_mode="none", transient_reward=0.25, termination_reward=1,
    transient_locations=[(6, 3), (4, 5), (2, 2)], transient_obstacles=[(4, 4)],
    agent_start_pos=(1, 1)
))

# plot env
env.env.render_mode = 'rgb_array'
state, _ = env.reset()
axes['a'].imshow(env.render())
axes['a'].set_xticks([])
axes['a'].set_yticks([])
env.env.render_mode = 'none'

# load agent
_, agent, env = load_trial(DATA_PATH / 'HealthyAgent_0.pkl')


results = []
heatmaps = {}
for rep in range(25):
    print('rep', rep)
    for noise_std, plot_panel in noise_levels.items():
        if plot_panel is not None:
            heatmaps[noise_std] = np.zeros((8,8))

        state, _ = env.reset()

        # add noise
        for layer in agent.policy_net:
            if hasattr(layer, 'std'):
                print(f'changing noise std from {layer.std} to {noise_std}')
                layer.std = noise_std

        for step in range(100):

            # record position
            if plot_panel is not None:
                heatmaps[noise_std][env.unwrapped.agent_pos[0], env.unwrapped.agent_pos[1]] += 1

            # get & execute action
            action = agent.select_action(state)
            new_state, reward, done, truncated, _ = env.step(action)

            if truncated or done:
                results.append({
                    'agent': agent.display_name,
                    'rep': rep,
                    'noise_std': noise_std,
                    'steps': step
                })

                break
            else:
                state = new_state

# generate plots
# noise_levels = {noise_std: panel for noise_std, panel in noise_levels.items() if noise_std in heatmaps}
max_steps = np.dstack(list(heatmaps.values())).max()

for noise_std, map in heatmaps.items():
    axes[noise_levels[noise_std]].imshow(map, vmin=0, vmax=max_steps, cmap='hot')
    axes[noise_levels[noise_std]].set_xticks([])
    axes[noise_levels[noise_std]].set_yticks([])
    axes[noise_levels[noise_std]].set_title(f'noise std={noise_std}')

plt.sca(axes['b'])
sns.lineplot(pd.DataFrame(results), x='noise_std', y='steps')
plt.ylabel('steps to reach goal')
plt.xlabel('standard deviation of added noise')

# add labels
for label, ax in axes.items():
    # label physical distance to the left and up:
    trans = mtransforms.ScaledTranslation(-20 / 72, 7 / 72, fig.dpi_scale_trans)
    ax.text(
        0.0,
        1.0,
        label,
        transform=ax.transAxes + trans,
        fontsize="large",
        weight="bold",
        va="bottom",
        fontfamily="serif",
    )

# add colorbar
normalizer = Normalize(0, max_steps)
im = cm.ScalarMappable(norm=normalizer, cmap='hot')
fig.colorbar(im, ax=[axes['c'], axes['d'], axes['e']])

# plt.tight_layout()
print(max_steps)
plt.show()
exit()

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
