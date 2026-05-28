from blue_ai.scripts.train_agents import load_dataset
import pandas as pd
pd.options.display.width = None
import numpy as np
import matplotlib.pyplot as plt
from blue_ai.envs.transient_goals import TransientGoals
import matplotlib.colors as mcolors
from skimage.transform import resize

interpolation_order = 0  # <--- try 0, 1, 2, maybe 3. Gives different visual effects when interpolating the heatmap
gamma_correction = 0.5  # <--- smaller numbers make hotspots in the heatmap less dramatic. Bigger numbers make them more dramatic
env_weight = 0  # <--- adjust between 0-1. Smaller makes the env fainter in the final image

# trials for which we want heatmaps for
trial_types = ['_testing',
               '_traumatized_testing',
               '_exposure_therapy_2_testing',
               '_connectivity_restoration_testing']
data_sets = []



for i in range(len(trial_types)):
    trial_type = trial_types[i]
    data = load_dataset(
        [
            # f'PTSDAgent_{trial}{trial_type}.pkl'  # <--- set desired filename pattern here
            f'TraumaSynapticDeficitAgent_{trial}{trial_type}.pkl'  # <--- set desired filename pattern here
            for trial in range(0, 30)
        ]
    )
    print('filenames loaded:\n',data['filename'].unique())
    data_sets.append(data)


# get an image of the environment
env = TransientGoals(
                    render_mode="rgb_array",
                    n_transient_obstacles = 1, n_transient_goals=3, agent_start_dir=1,
                    wall_locations =[[3,2],[3,3],[3,4],[3,5]], #[3,6]], # <--- I thought there was a doorway at the bottom of the env, but this is the wall setup I'm seeing in Benito's code
                    highlight=False
                )
env.reset()
env_rgb = env.render() / 256

# generate the heatmap for each trial
heatmaps = []
for data in data_sets:
    # get counts of the agent's state occupancy
    heatmap = np.zeros((8, 8))
    for position in data['position']:
        heatmap[position[1], position[0]] += 1
    heatmaps.append(heatmap)

# seeing if there is a bottom door present in each trial
for heatmap in heatmaps:
    print("Number of moves through the (potential) door: ", heatmap[6][3]) # <--- does appear to be a bottom door, as any non-zero value indicates movement

# find the absolute min/max for consistent heatmap coloring across trials
abs_heatmap_max = np.max(heatmaps)
abs_heatmap_min = np.min(heatmaps)
# print(abs_heatmap_max, abs_heatmap_min) <--- see the 'hottest' and 'coolest' value
# print(data_sets[0]['position'].shape[0]) <--- potential max/'hottest' value. this would only happen if the agent never moved

# create custom heatmap color range to match poster color scheme
colors = [(0,0,0), (41/255, 96/255, 145/255), (255/255, 255/255, 255/255)] # <--- color range. numerator is rbg values, denominator normalizes
cmap_name = 'poster_blue'

# convert counts to heatmap
# cmap = plt.get_cmap('bone') <--- if we want to use a default color scheme instead like 'bone' or 'hot'
cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
norm = mcolors.Normalize(vmin=abs_heatmap_min, vmax=abs_heatmap_max)  # <---- you may need to hardcode the heatmap min/max to get consistent meaning of colors across visuals (otherwise the brightest color in each visual will represent the biggest count in that particular dataset)
# color the heatmaps
for i in range(len(heatmaps)):
    heatmaps[i] = cmap(norm(heatmaps[i]))[:, :, :3]
# heatmap = cmap(norm(heatmap))[:, :, :3]

titles = ['before trauma', 'after trauma', 'after "exposure therapy"', 'after relieving connectivity deficits']

heatmap_no = 0 #<--- for labeling the outputs/indexing the trial_types list
for i in range(len(heatmaps)):
    heatmap = heatmaps[i]
    # resize heatmap to match size of environment
    heatmap = resize(heatmap, (256, 256), order=interpolation_order, anti_aliasing=False)
    heatmap /= heatmap.max()
    heatmap **= gamma_correction

    # plot
    plt.subplot(1, 4, i+1)
    plt.imshow(heatmap * (1 - env_weight) + env_rgb * env_weight, cmap=cmap)
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
    # plt.title(trial_types[heatmap_no])
    # plt.colorbar()
    heatmap_no += 1

plt.show()

# plot
plt.imshow(env_rgb, cmap=cmap)
plt.xticks([])
plt.yticks([])
plt.title("env")
# plt.colorbar()
plt.show()
