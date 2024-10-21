import pickle
import numpy as np
import pandas as pd  # Use Pandas instead of Polars
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from blue_ai.scripts.constants import DATA_PATH

def open_file(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def main():
    id = "3"
    all_results = pd.DataFrame(columns=["step", "reward", "rep"])
    dropout = 0
    noise = ()
    phase_duration = []

    for rep in range(0, 5):

        #load data and trial variables
        filename = DATA_PATH / f"rehab_nr_{id}_trial_{rep}.pkl"
        # filename = DATA_PATH/ f"rehabilitate_{rep}.pkl"
        data = open_file(filename)
        dropout = data["weight_dropout"]
        noise = data["noise"]
        phase_duration = data["phase_durations"]
        rewards = data["results"]["rolling_avg_reward"]

        if isinstance(rewards, np.ndarray):
            rewards = rewards.tolist()

        #create step list
        step_values = np.arange(len(rewards), dtype=int).tolist()


        rep_row = pd.DataFrame({
            "step": step_values,
            "reward": rewards,
            "rep": [rep] * len(rewards)
        })

        all_results = pd.concat([all_results, rep_row], ignore_index=True)


    print(phase_duration)

    plt.style.use('seaborn-v0_8')
    ax = sns.lineplot(data=all_results, x="step", y="reward", errorbar="sd", color="darkred",  estimator="mean")
    plt.title(f"dropout: {dropout}, noise: {noise}")

    # color the different phases
    phase_line = 0
    cmap = cm.get_cmap('tab20c', len(phase_duration))
    phase_colors = [cmap(i) for i in range(len(phase_duration))]
    phase_names = ["healthy", "depressed", "entropic", "treated"]

    for i in range(len(phase_duration)-1):
        phase_line += phase_duration[i]
        if i == 0:
            ax.axvspan(xmin=0, xmax=phase_line, facecolor=phase_colors[i], alpha=0.3)

        else:
            ax.axvspan(xmin=phase_line, xmax=phase_line + phase_duration[i+1], facecolor=phase_colors[i], alpha=0.3)
        print(phase_line, phase_line + phase_duration[i+1])

    plt.show()

if __name__ == '__main__':
    main()
