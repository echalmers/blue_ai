from blue_ai.scripts.constants import DATA_PATH, _CURRENT_DIR, N_TRIALS
import matplotlib.pyplot as plt
from blue_ai.scripts.view_performance import PerformancePlotter
from blue_ai.agents.agent_classes import HealthyAgent, SchizophrenicAgent
from blue_ai.scripts.train_agents import load_trial, load_dataset
import seaborn as sns
import pandas as pd


mosaic = """
    bc
    de
    """
fig, axes = plt.subplot_mosaic(
    mosaic,
    figsize=(11, 8),
    # width_ratios=[10, 10],
    # height_ratios=[47.5, 5, 47.5],
)


result_sets = [
    ('healthy before noise', 'b', 'HealthyAgent_*.pkl', True),
    ('healthy after noise', 'd', 'addNoise_HealthyAgent_*.pkl', False),
    ('E/I imbalance before noise', 'c', 'SchizophrenicAgent_*.pkl', False),
    ('E/I imbalance after noise', 'e', 'addNoise_SchizophrenicAgent_*.pkl', False),
]

for result_set in result_sets:
    df = load_dataset(filename_patterns=result_set[2])

    plt.yticks([0, 0.5, 1, 1.5, 2])
    plt.grid()

    plotter = PerformancePlotter(results_dataframe=df)
    plotter.plot_goals_per_episode(axes[result_set[1]], last_n_steps=10_000)

    plt.title(result_set[0])
    plt.xticks([])
    plt.xlabel('')
    plt.ylim([0, 2.2])

    if result_set[1] in ['b', 'd']:
        plt.ylabel('number of objects found per episode')
    else:
        y_ticks = plt.yticks()[0]
        plt.yticks(y_ticks, labels=['' for _ in range(len(y_ticks))])

    if result_set[3]:
        # sns.move_legend()
        pass
    else:
        plt.legend([], [], frameon=False)


plt.tight_layout()
plt.savefig(_CURRENT_DIR / 'schizophrenia' / 'img' / 'effect_of_noise.png', dpi=400)
plt.show()
