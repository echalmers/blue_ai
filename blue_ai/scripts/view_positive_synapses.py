from matplotlib.axes import Axes
from numpy.random import randint
from blue_ai.scripts.constants import FIGURE_PATH
import blue_ai.scripts.train_agents

import seaborn as sns
import matplotlib.pyplot as plt
from functools import lru_cache

from warnings import warn
import itertools


def flatten(it):

    return itertools.chain.from_iterable(it)


def remove_legend(*args, _debug=False):
    """
    Remove the legend if it is present
    """
    for ax in args:
        if legend := ax.get_legend():
            legend.remove()


def get_agent_names(x):
    return x["agent_name"].unique()


def plot_data(data, colors, line_offset=0):
    if len(data) <= 0:
        warn("No data passed; Skipping")
        return

    ax1 = plt.subplot(2, 3, 1 + line_offset)
    sns.lineplot(
        data=data,
        y="%pos_synapse",
        x="step",
        hue="agent_name",
        palette=colors,
    )

    ax1.set_ylabel("% Positive Synapses")
    remove_legend(ax1)

    ax2 = plt.subplot(2, 3, 2 + line_offset)
    sns.lineplot(
        data=data,
        y="mean_synapse",
        x="step",
        hue="agent_name",
        palette=colors,
    )
    remove_legend(ax2)

    ax3 = plt.subplot(2, 3, 3 + line_offset)
    sns.lineplot(
        data=data,
        y="cumulative_reward",
        x="step",
        hue="agent_name",
        palette=colors,
    )
    remove_legend(ax3)

    handles, labels = ax1.get_legend_handles_labels()

    return handles, labels


@lru_cache(maxsize=None)
def get_alpha(x: str):
    """
    >>> get_alpha("/scripts/data/PositiveLossAgent_0.03125_0.pkl")
    0.03125
    >>> get_alpha("/blue_ai/scripts/data/PositiveLossAgent_0.5_0.pkl")
    0.5
    >>> get_alpha("/blue_ai/blue_ai/scripts/data/PositiveLossAgent_5_0.pkl")
    5.0
    """
    from re import search

    PATTERN = r"PositiveLossAgent_([0-9]*\.?[0-9]+)_[0-9]+"
    match = search(PATTERN, str(x))

    # Return None or the found match
    return match and float(match.group(1))


if __name__ == "__main__":
    plt.figure(figsize=(6 * 3, 6 * 2))  # Set the figure size

    data = blue_ai.scripts.train_agents.load_dataset("*.pkl")

    data = data[(data["step"] % 10_000 == 0)]

    data["%pos_synapse"] = (data["num_pos_synapse"] / 10).rolling(20).mean()
    data["alpha"] = data["filename"].apply(get_alpha)

    data["agent_name"] = data["agent"] + data["alpha"].apply(
        lambda x: x and f"-{x}α" or ""  # I'm sorry
    )

    # Agents without an "alpha"
    non_sweeping = data[(data["alpha"].isna())]
    sweeping = data[(data["alpha"].notna())]

    non_sweeping_keys = get_agent_names(non_sweeping)
    sweeping_keys = get_agent_names(sweeping)

    keys = [
        (non_sweeping_keys, sns.color_palette("Set1")),
        (sweeping_keys, sns.color_palette("flare")),
    ]

    colors = dict(flatten(zip(x[0], x[1]) for x in keys))

    if result := plot_data(data[(data["alpha"].isna())], colors):
        h, l = result
        plt.figlegend(h, l, loc="upper center", ncol=3, title="Agent")

    if result := plot_data(data[(data["alpha"].notna())], colors, line_offset=3):
        h, l = result
        plt.figlegend(h, l, loc="lower center", ncol=3, title="Agent")

    plt.savefig(FIGURE_PATH / "positive.png")
