import os
from pathlib import Path

import holoviews as hv
import numpy as np
import polars as pl
import xxhash
from holoviews.operation.datashader import datashade
from hvplot.polars import hvPlotTabularPolars as plot
from numpy.typing import ArrayLike

from blue_ai.scripts.constants import DATA_PATH, FIGURE_PATH

hv.extension("bokeh")


def save_if_not_exists(graph, filepath):
    if not filepath.exists() or "REDO" in os.environ:
        hv.save(graph, filepath)

    else:
        print(f"Skipping saving {filepath}")


def fill_at_indices(indices: ArrayLike, values: ArrayLike, *_, length=None):
    length = length or np.array(indices).max() + 1
    z = np.zeros(length, dtype=values.dtype)
    z[indices] = values
    return z


def main():
    disk_data = pl.scan_parquet(DATA_PATH / "ratios.parquet")

    hash = xxhash.xxh32()
    data = (
        disk_data.select(~pl.selectors.by_name("params"))
        .with_columns(
            stage=pl.col("state").alias("stage"),
            ratio=pl.col("ratio_reward") / pl.col("ratio_penalty"),
        )
        .collect()
    )
    hash.update(data.hash_rows().to_numpy().tobytes())

    data = data.with_columns(
        id=pl.struct(["agent", "stage", "ratio_reward"]),
    )

    def path(path):
        p = path if type(path) is Path else Path(path)
        ext = "".join(p.suffixes)
        name = str(p).rstrip(ext)

        return FIGURE_PATH / f"{name}_{hash.hexdigest()}{ext}"

    heatmap = (
        plot(data)
        .points(
            x="agent_pos_x",
            y="agent_pos_y",
            datashade=True,
            aggregator="count",
            cmap="inferno",
            colorbar=False,
            xaxis=False,
            yaxis=False,
            row="ratio_reward",
            col="agent",
            groupby=["episode"],
        )
        .opts(
            width=200,
            height=200,
        )
    )
    # hvplot.show(heatmap)
    save_if_not_exists(heatmap, path("heat_map_explore.gif"))

    dist_entropy = (
        pl.struct(["agent_pos_x", "agent_pos_y"])
        .value_counts()
        .map_elements(
            lambda grid: grid.struct["count"].entropy(),
            return_dtype=pl.Float64,
        )
    )

    exploration = plot(
        data.group_by("id", "episode", "trial_id")
        .agg(step=pl.col("step").mean(), entropy=dist_entropy)
        .sort("step")
        .with_columns(pl.col("entropy").rolling_mean(100, min_periods=1).over("id"))
        .unnest("id")
        .sort("ratio_reward")
    ).line(
        x="step",
        y="entropy",
        label="Agent Per Episode Exploration Rate, Higher is better",
        by=["ratio_reward"],
        groupby="agent",
    )

    # save_if_not_exists((heatmap + exploration).cols(1), path("combined.html"))

    save_if_not_exists(exploration, path("exploration.html"))


if __name__ == "__main__":
    main()
