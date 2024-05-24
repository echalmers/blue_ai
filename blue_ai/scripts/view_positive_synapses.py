from blue_ai.scripts.constants import FIGURE_PATH
import blue_ai.scripts.train_agents

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = blue_ai.scripts.train_agents.load_dataset("*.pkl")
    plt.figure(figsize=(6 * 3, 6))  # Set the figure size

    data["%pos_synapse"] = np.where(
        data["num_pos_synapse"] != 0, data["num_pos_synapse"] / 10, 0
    )
    data["%pos_synapse"] = data["%pos_synapse"].rolling(1000).mean()

    data = data[(data["step"] % 20 == 0)]

    ax1 = plt.subplot(1, 3, 1)
    sns.lineplot(
        data=data,
        y="%pos_synapse",
        x="step",
        hue="agent",
    )

    ax1.set_ylabel("% Positive Synapses")
    ax1.get_legend().remove()

    ax2 = plt.subplot(1, 3, 2)
    sns.lineplot(
        data=data,
        y="mean_synapse",
        x="step",
        hue="agent",
    )
    ax2.get_legend().remove()

    ax3 = plt.subplot(1, 3, 3)
    sns.lineplot(
        data=data,
        y="cumulative_reward",
        x="step",
        hue="agent",
    )
    ax3.get_legend().remove()

    handles, labels = ax2.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc="upper center", ncol=3, title="Agent")

    plt.savefig(FIGURE_PATH / "positive.png")


if __name__ == "__main__":
    main()