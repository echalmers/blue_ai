import seaborn as sns
import matplotlib.pyplot as plt
from constants import DATA_PATH, FIGURE_PATH
import pandas as pd

import numpy as np


def rotate_about_point(point, radians=None, degrees=None, origin=np.array([0, 0])):
    point, origin = np.array(point), np.array(origin)

    assert bool(radians) ^ bool(degrees), "Provide exactly one of radians or degrees"

    if not radians:
        radians = np.radians(degrees)

    rot = np.array(
        [
            [np.cos(radians), -1 * np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ]
    )

    return (rot @ np.array(point - origin)) + origin


def main():
    data = pd.read_parquet(DATA_PATH / "branching.parquet")


    g = sns.FacetGrid(data[data["path"].str.len() > 1], row="path", col="agent")

    origin = [
        data["agent_pos_x"].unique().mean(),
        data["agent_pos_y"].unique().mean(),
    ]
    positions = data[["agent_pos_x", "agent_pos_y"]].to_numpy()

    rotated_positions = np.apply_along_axis(
        rotate_about_point, 1, positions, degrees=180, origin=origin
    )

    data["agent_pos_x"] = rotated_positions[:, 0].astype(int)
    data["agent_pos_y"] = rotated_positions[:, 1].astype(int)

    g.map_dataframe(
        sns.histplot,
        x="agent_pos_x",
        y="agent_pos_y",
        discrete=True,
        stat="density",
        pthresh=0,
    )

    plt.savefig(FIGURE_PATH / "agent_position_distribution.png")


if __name__ == "__main__":
    main()
