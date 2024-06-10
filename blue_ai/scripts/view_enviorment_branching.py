from numpy import cumsum
import pandas as pd
from blue_ai.scripts.constants import DATA_PATH, FIGURE_PATH

import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from blue_ai.scripts.view_positive_synapses import remove_legend
from blue_ai.scripts.enviorment_branching import STEPS_PER_STAGE


def load():
    with open(DATA_PATH / "branching.pkl", "rb") as f:
        return pickle.load(f)


def main():
    plt.figure(figsize=(6 * 4, 6 * 1 + 2))  # Set the figure size

    # Add Vertical lines indicating stage transition
    for threshold in cumsum(STEPS_PER_STAGE):
        plt.axvline(threshold)

    data: pd.DataFrame = load()

    data["possible_reward"] = data.groupby(["agent", "path", "trial_id"])[
        "total_reward"
    ].transform("sum")

    data["cumulative_reward"] = (
        data["cumulative_reward"] / data["possible_reward"]
    ) * 100

    keys = data["path"].unique()
    keys = sorted(keys, key=(lambda x: x.count("B") - x.count("G")))
    colors = sns.diverging_palette(250, 0, l=65, center="dark", n=len(keys))
    palette = dict(list(zip(keys, colors)))

    data = data[(data["step"] % 100 == 0)]

    ax1 = sns.lineplot(
        data=data,
        x="step",
        y="cumulative_reward",
        hue="path",
        style="agent",
        palette=palette,
        # estimator=None,
        # n_boot=0,
    )

    ax1.set_ylabel("% Reward Obtained")

    ax1.set_xlabel("Total Steps")

    h, l = ax1.get_legend_handles_labels()
    plt.figlegend(h, l, loc="upper center", ncols=len(keys) // 2)
    remove_legend(ax1)

    plt.savefig(FIGURE_PATH / "branching.png")


if __name__ == "__main__":
    main()
