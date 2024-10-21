# Paths to the files
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.cm as cm
from blue_ai.scripts.constants import DATA_PATH


def main():
    id = "3"
    stages = ["healthy", "depressed", "entropic", "treated"]
    all_trials_data = pd.DataFrame()
    stages_length = {}
    for i in range(0, 5):
        trial_data = pd.DataFrame()
        for stage in stages:
            stage_data = pd.read_parquet(DATA_PATH / f"{id}_{stage}_{i}_sliced_entropies.parquet")
            stage_data["trial_nr"] = i
            stage_data["stage"] = stage

            trial_data = pd.concat([trial_data, stage_data])
            trial_data["time_point"] = list(range(len(trial_data)))

            if stage not in stages_length:
                stages_length[stage] = len(stage_data)

        all_trials_data = pd.concat([all_trials_data, trial_data])

    print(stages_length)
    #plotting
    plt.style.use('seaborn-v0_8')
    ax = sns.lineplot(data=all_trials_data, x="time_point", y="1", errorbar="sd", estimator="mean")

    #color different phases
    cmap = cm.get_cmap('tab20c', len(stages_length))
    phase_colors = [cmap(i) for i in range(len(stages_length))]
    xmin = 0

    for i, stage in enumerate(stages):
        xmax = xmin + stages_length[stage]
        ax.axvspan(xmin=xmin, xmax=xmax, facecolor=phase_colors[i], alpha=0.3)
        xmin = xmax


    plt.title(f"connectivity entropy throughout the rehab stages. Id: {id}")
    plt.xlabel("Time Point")
    plt.ylabel("Entropy")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
