import os
from pathlib import Path

import holoviews as hv
import numpy as np
import polars as pl
import xxhash
from hvplot.polars import hvPlotTabularPolars as plot
from blue_ai.scripts.constants import DATA_PATH, FIGURE_PATH

# import matplotlib
# hv.extension("matplotlib")
# matplotlib.use("agg")


def save_if_not_exists(graph, filepath):
    if not filepath.exists() or "REDO" in os.environ:
        print(hv.save(graph, filepath))
        print(filepath.absolute().as_uri())
    else:
        print(f"Skipping saving {filepath}")


def list_to_number(digit_list):
    place_values = 10 ** np.arange(len(digit_list) - 1, -1, -1)
    number = np.dot(digit_list, place_values)

    return number


def heatmap(data: pl.DataFrame):
    heatmap_args = dict(
        x="agent_pos_x",
        y="agent_pos_y",
        datashade=True,
        rasterize=True,
        width=data.select(pl.col("agent_pos_x").max()).item(),
        height=data.select(pl.col("agent_pos_y").max()).item(),
        rot=90,
        xaxis=False,
        yaxis=False,
        aggregator="count",
        col="agent",
        row="ratio_reward",
        responsive=True,
        cmap="inferno",
    )

    stage0 = plot(data.filter(pl.col("stage") == 0)).points(
        label="Stage 0", **heatmap_args
    )

    groups = (
        data.filter(pl.col("stage") == 1)
        .sort("stage_0_reward")
        .group_by(["stage_0_reward"], maintain_order=True)
    )

    stage1s = []
    for key, g in groups:
        stage1s.append(plot(g).points(label=f"Stage 1,from {key[0]}", **heatmap_args))

    return (
        (hv.Layout(stage1s).opts(merge_tools=True) + stage0)
        .cols(len(stage1s))
        .opts(tabs=True)
    )


@pl.StringCache()
def main():
    disk_data = pl.scan_parquet(DATA_PATH / "full.parquet")

    hash = xxhash.xxh32()
    max_width = disk_data.select(pl.col("ratios").list.len().max()).collect().item()

    data = disk_data.with_columns(
        *[
            (
                pl.col("ratios").list.get(x).cast(pl.String)
                + ":"
                + pl.col("ratios").list.get(x + 1).cast(pl.String)
            )
            .cast(pl.Categorical)
            .alias(f"stage_{x // 2}_ratio")
            for x in range(0, max_width, 2)
        ],
        *[
            pl.col("ratios").list.get(x).alias(f"stage_{x // 2}_reward")
            for x in range(0, max_width, 2)
        ],
        *[
            pl.col("ratios").list.get(x + 1).alias(f"stage_{x // 2}_penalty")
            for x in range(0, max_width, 2)
        ],
        ratio=pl.col("ratio_reward") / pl.col("ratio_penalty"),
        str_ratios=pl.col("ratios")
        .list.eval(pl.element().cast(pl.String))
        .list.join("-"),
    ).drop("ratios")

    def path(path):
        p = path if type(path) is Path else Path(path)
        ext = "".join(p.suffixes)
        name = str(p).rstrip(ext)
        return FIGURE_PATH / f"{name}_{hash.hexdigest()}{ext}"

    hash.update(data.head(100).collect().hash_rows().to_numpy().tobytes())

    id_select = pl.col("agent"), pl.selectors.matches("stage_[0-9]+_(reward|penalty)")
    data = data.with_columns(id=pl.struct(*id_select))

    by_trial = (
        data.group_by(
            "agent",
            "stage_0_ratio",
            "stage_1_ratio",
            "trial_id",
            "stage",
        )
        .agg(
            pl.col("spine_loss"),
            (pl.col("reward").cum_sum() / pl.col("total_reward").cum_sum())
            .alias("util")
            .rolling_mean(200),
            pl.col("step"),
        )
        .explode("util", "step", "spine_loss")
    )

    util = (
        by_trial.sort("agent", "stage_0_ratio", "stage_1_ratio", "step")
        .filter(pl.col("util").is_finite())
        .with_columns(
            util=pl.col("util")
            .mean()
            .over("agent", "stage_0_ratio", "stage_1_ratio", "step"),
            agent=pl.col("agent").cast(pl.String),
        )
    )

    breakpoint()

    utilization = plot(util).line(
        x="step",
        y="util",
        by="agent",
        row="stage_0_ratio",
        col="stage_1_ratio",
        ylim=(-1, 1),
        datashade=True,
        rasterize=True,
        line_width=5,
    )

    save_if_not_exists(utilization, path("util.html"))


if __name__ == "__main__":
    main()
