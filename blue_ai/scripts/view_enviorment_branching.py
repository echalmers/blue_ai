from typing import Any, Dict, List
from matplotlib.axes import Axes
from blue_ai.scripts.constants import DATA_PATH, FIGURE_PATH
from blue_ai.scripts.enviorment_branching import STEPS_PER_STAGE
from blue_ai.scripts.view_positive_synapses import remove_legend

from itertools import product
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import torch

from tqdm import tqdm

import pickle

# Minimum Guarantee of points in a given series, we will amplify the amount around points of interest later
POINTS_PER_SERIES = 100

# To get a unique environment
UNIQUE_RUNS = ["path", "trial_id", "agent"]

transiation_centers = np.cumsum(STEPS_PER_STAGE)


def load():
    with open(DATA_PATH / "branching.pkl", "rb") as f:
        return pickle.load(f)


def postfix_elements(elements):
    count_dict: Dict[str, int] = {}
    result = []

    for elem in elements:
        if elem in count_dict:
            count_dict[elem] += 1
        else:
            count_dict[elem] = 1
        result.append(f"{elem}_{count_dict[elem]}")

    return result


def create_subplots(height, width):
    # Setup plot layout
    fig, axes = plt.subplots(height, width, sharex=True, sharey="row")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    fig.set_figwidth(6 * 4)
    fig.set_figheight(4 * height)

    return fig, axes


def graph_cols(cols, axes, data: pd.DataFrame, palette):
    for i, col in enumerate(cols):
        ax = axes[i]

        sns.lineplot(
            data=data,
            x="step",
            y=f"{col}_mean",
            hue="path",
            ax=ax,
            estimator=None,  # Disable the built in esitamtor, it is so stupidly slow it's incredible
            palette=palette,
        )

        ax.set_ylabel(col)

        # The fact that this is orders of magnitude faster than what seaborn
        # does is baffeling. Because we aggregate the data ahead of time we don't
        # need to call groupby in a for loop which tends to help with perormance
        for id in data["path"].unique():
            d = data[data["path"] == id]

            ax.fill_between(
                d["step"],
                d[f"{col}_mean"] + d[f"{col}_std"],
                d[f"{col}_mean"] - d[f"{col}_std"],
                color=palette[id],
                alpha=0.1,
            )


def main():
    data: pd.DataFrame = pd.read_parquet(DATA_PATH / "no_layers.parquet")

    # data.path = pd.Categorical(data["path"].str[::-1])

    GRAPHED_COLUMNS = [
        "cumulative_reward",
        "% Utilization",
        "Reward Per Step",
    ]

    agents = data["agent"].unique()
    fig, axes = create_subplots(len(GRAPHED_COLUMNS), len(agents))

    data.sort_values(UNIQUE_RUNS, inplace=True)

    if axes.ndim == 1:
        axes = axes.reshape((len(axes), 1))

    # Add a line at each transion center
    ax: Axes
    for ax, threshold in product(np.ndarray.flatten(axes), transiation_centers):
        ax.axvline(threshold)

    cos = torch.nn.CosineSimilarity(dim=1)

    runs = data.groupby(UNIQUE_RUNS, observed=True)

    def cosine_distances(x):
        t = torch.from_numpy(np.array(x.to_numpy().tolist())).flatten(start_dim=1)

        results = torch.arccos(cos(t[1:], t[:-1]))

        return np.cumsum(np.pad(results, (0, 1), constant_values=np.nan))

    for col in [c for c in GRAPHED_COLUMNS if "layer_" in c]:
        data[col] = runs[col].transform(cosine_distances)

    data["Reward Per Step"] = runs["reward"].transform(lambda x: x.rolling(2000).mean())

    data["% Utilization"] = 100 * (
        (runs["reward"].cumsum() / runs["total_reward"].cumsum())
    )

    palette = dict(list(zip(data["path"].unique(), sns.color_palette())))

    # The types of data we want about each step
    aggregate_types = ["min", "max", "std", "mean"]

    # Exclude outliers from the graph coming from div by zero or other floating
    # point errors that pop up near the beginning of the sample

    for ax in axes.transpose():
        ax[1].set_ylim(
            IQR(data["% Utilization"]).min(), IQR(data["% Utilization"]).max()
        )

    data = (
        data.groupby(["agent", "path", "step"], observed=True)
        # We need to used named columns in order to not use multiindexing which
        # neither I nor seaborn seems to play nice with
        .aggregate(
            **{
                f"{col_name}_{agg}": (col_name, agg)
                for col_name, agg in product(GRAPHED_COLUMNS, aggregate_types)
            }
        ).reset_index()
    )

    hs, ls = [], []

    for a, ax in zip(agents, axes.transpose()):
        top: Axes = ax[0]

        top.set_title(a)

        graph_cols(
            GRAPHED_COLUMNS,
            ax,
            data[data["agent"] == a],
            palette,
        )

        # remove_legend(*ax)

        h, layers = top.get_legend_handles_labels()

        hs.append(h)
        ls.append(layers)

    # Create a non repeating legend
    h, layers = np.array(hs).flatten(), np.array(ls).flatten()

    legend = dict(zip(layers, h))

    h = list(legend.values())
    layers = list(legend.keys())

    # plt.figlegend(h, layers, loc="upper center", ncols=len(h))

    plt.savefig(FIGURE_PATH / "branching.png")


# Thanks Chatgpt!
def IQR(data):
    """
    Returns the middle of the Interquartile range
    See : https://en.wikipedia.org/wiki/Interquartile_range
    """

    values = data.replace([np.inf, -np.inf], np.nan).dropna()

    R = 10

    Q1 = np.percentile(values, R)
    Q3 = np.percentile(values, 100 - R)

    # Calculate IQR
    IQR = Q3 - Q1

    # Identify outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return data[(data >= lower_bound) & (data <= upper_bound)]


if __name__ == "__main__":
    main()
