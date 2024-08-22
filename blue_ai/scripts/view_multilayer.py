import os
from pathlib import Path
import holoviews as hv
import polars as pl
import xxhash
from hvplot.polars import hvPlotTabularPolars as plot
from blue_ai.scripts.constants import DATA_PATH, FIGURE_PATH


def save_if_not_exists(graph, filepath):
    if not filepath.exists() or "REDO" in os.environ:
        print(hv.save(graph, filepath))
        print(filepath.absolute().as_uri())
    else:
        print(f"Skipping saving {filepath}")


def heatmap(data: pl.DataFrame):
    """
    Takes a polars dataframe expects to following columns:
        - position: agent_pos_x, agent_pos_y
        - category: ratio_reward, agent
    The heat map will be calculated based on the count of unique (x,y) pairs
    """
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


# Enable string caching, this allows for the string manips we do later on to
# not overflow memory or drasticly slow us down
@pl.StringCache()
def main():
    disk_data = pl.scan_parquet(DATA_PATH / "full.parquet")
    # Create a hash instance to be used for indefying this dataset
    hash = xxhash.xxh32()
    # Get the widest amount sub array of ratios
    max_width = disk_data.select(pl.col("ratios").list.len().max()).collect().item()

    # Break the ratios in to more useful columns for graphing
    data = disk_data.with_columns(
        # A pair of Good:Bad for the each stage
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
        # Break the rewards into its own column
        *[
            pl.col("ratios").list.get(x).alias(f"stage_{x // 2}_reward")
            for x in range(0, max_width, 2)
        ],
        # Break the lavas into its own column
        *[
            pl.col("ratios").list.get(x + 1).alias(f"stage_{x // 2}_penalty")
            for x in range(0, max_width, 2)
        ],
        ratio=pl.col("ratio_reward") / pl.col("ratio_penalty"),
        # '-' separated list of all the ratios for each line
        str_ratios=pl.col("ratios")
        .list.eval(pl.element().cast(pl.String))
        .list.join("-"),
    ).drop("ratios")

    hash.update(data.head(100).collect().hash_rows().to_numpy().tobytes())

    def path(path):
        p = path if type(path) is Path else Path(path)
        ext = "".join(p.suffixes)
        name = str(p).rstrip(ext)
        return FIGURE_PATH / f"{name}_{hash.hexdigest()}{ext}"

    util = pl.col("reward").cum_sum() / pl.col("total_reward").cum_sum()

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
            pl.col("step"),
            util=util,
        )
        .explode("util", "step", "spine_loss")
    )

    breakpoint()


if __name__ == "__main__":
    main()
