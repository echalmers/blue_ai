import re
from blue_ai.scripts.constants import FIGURE_PATH
import blue_ai.scripts.train_agents

import seaborn as sns
import matplotlib.pyplot as plt
from functools import lru_cache


def plot_data(data, colors, line_offset=0):
    ax1 = plt.subplot(2, 3, 1 + line_offset)
    sns.lineplot(
        data=data,
        y="%pos_synapse",
        x="step",
        hue="agent_name",
        palette=colors,
    )

    ax1.set_ylabel("% Positive Synapses")
    ax1.get_legend().remove()

    ax2 = plt.subplot(2, 3, 2 + line_offset)
    sns.lineplot(
        data=data,
        y="mean_synapse",
        x="step",
        hue="agent_name",
        palette=colors,
    )
    ax2.get_legend().remove()

    ax3 = plt.subplot(2, 3, 3 + line_offset)
    sns.lineplot(
        data=data,
        y="cumulative_reward",
        x="step",
        hue="agent_name",
        palette=colors,
    )
    ax3.get_legend().remove()

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
    match = re.search(r"PositiveLossAgent_([0-9]*\.?[0-9]+)_[0-9]+", str(x))

    if match is None:
        return None

    return match.group(1)


if __name__ == "__main__":
    plt.figure(figsize=(6 * 3, 6 * 2))  # Set the figure size

    data = blue_ai.scripts.train_agents.load_dataset("*.pkl")

    data = data[(data["step"] % 20 == 0)]

    data["%pos_synapse"] = (data["num_pos_synapse"] / 10).rolling(20).mean()
    data["alpha"] = data["filename"].apply(get_alpha)

    data["agent_name"] = data["agent"] + data["alpha"].apply(
        lambda x: x and f"-{x}α" or ""  # I'm sorry
    )

    non_sweeping_keys = data[(data["alpha"].isna())]["agent_name"].unique()
    sweeping_keys = data[(data["alpha"].notna())]["agent_name"].unique()

    colors = dict(
        list(zip(non_sweeping_keys, sns.color_palette("Set1")))
        + list((zip(sweeping_keys, sns.color_palette("flare"))))
    )

    h, l = plot_data(data[(data["alpha"].isna())], colors)

    plt.figlegend(h, l, loc="upper center", ncol=3, title="Agent")

    h, l = plot_data(data[(data["alpha"].notna())], colors, line_offset=3)

    plt.figlegend(h, l, loc="lower center", ncol=3, title="Agent")

    plt.savefig(FIGURE_PATH / "positive.png")
