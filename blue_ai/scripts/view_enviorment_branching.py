from blue_ai.scripts.constants import DATA_PATH, FIGURE_PATH
from blue_ai.scripts.enviorment_branching import STEPS_PER_STAGE
from blue_ai.scripts.view_positive_synapses import remove_legend

from itertools import product
from numpy import cumsum
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pickle
import seaborn as sns


def load():
    with open(DATA_PATH / "branching.pkl", "rb") as f:
        return pickle.load(f)


def main():
    fig, axes = plt.subplots(3, 1)

    fig.set_figwidth(6 * 4)
    fig.set_figheight(6 * 3)

    for ax, threshold in product(axes, cumsum(STEPS_PER_STAGE) - STEPS_PER_STAGE[0]):
        ax.axvline(threshold)

    data: pd.DataFrame = load()

    # Convert categorical columns
    data["agent"] = data["agent"].astype("category")
    data["path"] = data["path"].astype("category")
    data["trial_id"] = data["trial_id"].astype("category")

    data = data.reset_index()

    data["rolling"] = (
        data.groupby(["path", "trial_id", "agent"], sort=False)["reward"]
        .rolling(1000)
        .mean()
    ).reset_index(drop=True)

    data = data[data["step"] % 50 == 0]

    keys = data["path"].unique()
    keys = sorted(keys, key=(lambda x: (x.count("B") - x.count("G") / len(x))))
    colors = sns.diverging_palette(250, 0, l=65, center="dark", n=len(keys))
    palette = dict(list(zip(keys, colors)))

    print("Cumulative: ", end="")
    sns.lineplot(
        data=data,
        y="cumulative_reward",
        x="step",
        style="agent",
        hue="path",
        palette=palette,
        ax=axes[0],
    )
    print("Done")

    print("Rolling :", end="")
    sns.lineplot(
        data=data,
        y="reward",
        x="step",
        hue="path",
        palette=palette,
        ax=axes[1],
    )
    print("Done")

    sns.lineplot(
        data=data,
        y="rolling",
        x="step",
        hue="path",
        palette=palette,
        ax=axes[2],
    )

    remove_legend(*axes)

    # Create custom Legend at top of screen always two rows long
    h, l = axes[0].get_legend_handles_labels()
    plt.figlegend(h, l, loc="upper center", ncols=len(keys) // 2)

    plt.savefig(FIGURE_PATH / "branching.png")


if __name__ == "__main__":
    data = load()

    main()
