from blue_ai.scripts.train_agents import load_dataset
import pandas as pd
pd.options.display.width = None
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from blue_ai.scripts.ptsd.view_performance import PerformancePlotter
from blue_ai.envs.transient_goals import TransientGoals
import matplotlib.colors as mcolors
from skimage.transform import resize


names_map = {'simulated ptsd': 'pre-existing deficit',
             'spine loss after trauma': 'post-trauma deficit'}
names_map_2 = {'pre-existing deficit': 'pre-existing\ndeficit',
             'post-trauma deficit': 'post-trauma\ndeficit'}

colors = ['grey', 'royalblue', 'midnightblue']
learning_curves_only = True


if learning_curves_only:
    f, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax = np.expand_dims(ax, 0)
else:
    f, ax = plt.subplots(2, 3, figsize=(12, 4))


# # _____Initial Learning_____________
filenames = []
filenames += [f'HealthyAgent_{trial}.pkl' for trial in range(0, 30)]
filenames += [f'PTSDAgent_{trial}.pkl' for trial in range(0, 30)]
filenames += [f'TraumaSynapticDeficitAgent_{trial}.pkl' for trial in range(0, 30)]
data = load_dataset(filenames)
# data = data[data['step'] % 10 == 0]
data['agent'] = data['agent'].replace(names_map)

plotter = PerformancePlotter(results_dataframe=data)

plotter.plot_learning_curves(ax[0, 0], errorbar='se', palette=colors)
ax[0, 0].legend().set_title("")
ax[0, 0].set_ylim([-100, 1100])
ax[0, 0].set_title('cumulative reward during initial learning...')
ax[0, 0].grid(axis='y', alpha=0.3, linewidth=0.5)
ax[0, 0].set_xticks(list(range(0, 40_001, 10_000)))

if not learning_curves_only:
    data['agent'] = data['agent'].replace(names_map_2)
    plotter.plot_goals_per_episode(ax[1, 0])
    ax[1, 0].legend().set_title("")
    ax[1, 0].set_ylim([0, 1.8])
    ax[1, 0].set_title('objects reached per episode, during initial learning...')



# _____Immediately after trauma_____________
filenames = []
filenames += [f'HealthyAgent_{trial}_traumatized_testing.pkl' for trial in range(0, 30)]
filenames += [f'PTSDAgent_{trial}_traumatized_testing.pkl' for trial in range(0, 30)]
filenames += [f'TraumaSynapticDeficitAgent_{trial}_traumatized_testing.pkl' for trial in range(0, 30)]
data = load_dataset(filenames)
data['step'] += 40_000
data['agent'] = data['agent'].replace(names_map)

plotter = PerformancePlotter(results_dataframe=data)

plotter.plot_learning_curves(ax[0, 1], errorbar='se', palette=colors)
ax[0, 1].set_ylim([-100, 1100])
ax[0, 1].set_title('...immediately after trauma...')
ax[0, 1].legend().set_title("")
ax[0, 1].legend_.remove()
ax[0, 1].set_xticks([40000, 45000])
ax[0, 1].set_xlabel('')
ax[0, 1].grid(axis='y', alpha=0.3, linewidth=0.5)

if not learning_curves_only:
    plotter.plot_goals_per_episode(ax[1, 1])
    ax[1, 1].legend().set_title("")
    ax[1, 1].set_ylim([0, 1.8])
    ax[1, 1].set_title('...immediately after trauma...')
    ax[1, 1].legend_.remove()



# _____Relearning_____________
filenames = []
filenames += [f'HealthyAgent_{trial}_relearned.pkl' for trial in range(0, 30)]
filenames += [f'PTSDAgent_{trial}_relearned.pkl' for trial in range(0, 30)]
filenames += [f'TraumaSynapticDeficitAgent_{trial}_relearned.pkl' for trial in range(0, 30)]
data = load_dataset(filenames)
data['step'] += 45_000
data['agent'] = data['agent'].replace(names_map)

plotter = PerformancePlotter(results_dataframe=data)

plotter.plot_learning_curves(ax[0, 2], errorbar='se', palette=colors)
ax[0, 2].set_ylim([-100, 1100])
ax[0, 2].set_title('...and after re-learning')
ax[0, 2].legend().set_title("")
# ax[0, 2].legend_.remove()
ax[0, 2].set_xlabel('')
ax[0, 2].grid(axis='y', alpha=0.3, linewidth=0.5)
ax[0, 2].set_xticks(list(range(45_000, 85_001, 10_000)))

if not learning_curves_only:
    data['agent'] = data['agent'].replace(names_map_2)
    plotter.plot_goals_per_episode(ax[1, 2], legend=False)
    ax[1, 2].set_ylim([0, 1.8])
    ax[1, 2].set_title('...and after re-learning')
    ax[1, 2].legend().set_title("")
    ax[1, 2].legend_.remove()


# general cleanup
for row in [0] if learning_curves_only else [0, 1]:
    for col in [1, 2]:
        ax[row, col].set_yticklabels(['' for _ in range(len(ax[row, col].get_yticklabels()))])



plt.tight_layout()
plt.show()
