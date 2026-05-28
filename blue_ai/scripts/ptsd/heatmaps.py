from blue_ai.scripts.train_agents import load_dataset
import pandas as pd
pd.options.display.width = None
import numpy as np
import matplotlib.pyplot as plt
from blue_ai.envs.transient_goals import TransientGoals
import matplotlib.colors as mcolors
from skimage.transform import resize

interpolation_order = 2  # <--- try 0, 1, 2, maybe 3. Gives different visual effects when interpolating the heatmap
gamma_correction = 0.9  # <--- smaller numbers make hotspots in the heatmap less dramatic. Bigger numbers make them more dramatic
env_weight = 0.2  # <--- adjust between 0-1. Smaller makes the env fainter in the final image

data = load_dataset(
    [
        f'HealthyAgent_{trial}_traumatized_testing.pkl'  # <--- set desired filename pattern here
        for trial in range(0, 30)
    ]
)
print('filenames loaded:\n',data['filename'].unique())


# get an image of the environment
env = TransientGoals(
                    render_mode="rgb_array",
                    n_transient_obstacles = 0, n_transient_goals=0, agent_start_dir=1,
                    wall_locations =[[3,2],[3,3],[3,4],[3,5],[3,6]], # <--- I thought there was a doorway at the bottom of the env, but this is the wall setup I'm seeing in Benito's code
                    highlight=False
                )
env.reset()
env_rgb = env.render() / 256

# get counts of the agent's state occupancy
heatmap = np.zeros((8, 8))
for position in data['position']:
    heatmap[position[1], position[0]] += 1

# convert counts to heatmap
cmap = plt.get_cmap('hot')
norm = mcolors.Normalize(vmin=heatmap.min(), vmax=heatmap.max())  # <---- you may need to hardcode the heatmap min/max to get consistent meaning of colors across visuals (otherwise the brightest color in each visual will represent the biggest count in that particular dataset)
heatmap = cmap(norm(heatmap))[:, :, :3]

# resize heatmap to match size of environment
heatmap = resize(heatmap, (256, 256), order=interpolation_order, anti_aliasing=False)
heatmap /= heatmap.max()
heatmap **= gamma_correction

# plot
plt.imshow(heatmap * (1 - env_weight) + env_rgb * env_weight, cmap='hot')
plt.xticks([])
plt.yticks([])
# plt.colorbar()
plt.show()
