from numpy.typing import ArrayLike
import polars as pl
from polars.dependencies import hvplot
import numpy as np
import holoviews as hv
import scipy

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from blue_ai.scripts.constants import DATA_PATH, FIGURE_PATH

hv.extension("bokeh")


def fill_at_indices(indices: ArrayLike, values: ArrayLike, *_, length=None):
    length = length or np.array(indices).max() + 1
    z = np.zeros(length, dtype=values.dtype)
    z[indices] = values
    return z


def main():
    disk_data = pl.scan_parquet(DATA_PATH / "ratios.parquet")

    data = disk_data.select(~pl.selectors.by_name("params")).collect()

    numeric_lists = pl.selectors.by_dtype([pl.List(T) for T in pl.NUMERIC_DTYPES])
    unique = ["agent", "trial_id", "name"]

    map_shape = (data["agent_pos_x"].max(), data["agent_pos_y"].max())

    data = data.with_columns(
        data.group_by(unique, maintain_order=True)
        .agg(
            pl.col("transient_goal", "lava", "terminal_goal")
            .sum()
            .name.prefix("cum_amount_"),
            rolling_reward=pl.col("reward").rolling_mean(2000),
            cumulative_reward=pl.col("reward").cum_sum(),
            stage=(pl.col("step") // 20_000),
            ratio=pl.col("ratio_reward") / pl.col("ratio_penalty"),
            utilization=(
                pl.col("reward").cum_sum() / pl.col("total_reward").cum_sum()
            ).rolling_mean(20),
            agent_cord=(pl.col("agent_pos_x") - 1)
            + (pl.col("agent_pos_y") - 1) * map_shape[0],
        )
        .explode(numeric_lists)
    )

    grouping = [
        "ratio_reward",
        "agent",
    ]

    dist = (
        data.group_by(*grouping, maintain_order=True)
        .agg(
            pl.col("agent_cord")
            .map_elements(
                lambda x: np.log1p(
                    (
                        map := fill_at_indices(
                            *np.unique(x, return_counts=True), length=36
                        )
                    )
                    / map.sum()
                ).tolist()
            )
            .alias("prob"),
            pl.lit((np.arange(0, 36) // 6).tolist()).alias("agent_pos_y"),
            pl.lit((np.arange(0, 36) % 6).tolist()).alias("agent_pos_x"),
        )
        .with_columns(entropy=pl.col("prob").map_elements(scipy.stats.entropy))
        .explode(numeric_lists)
    )

    data = (
        data.group_by(*grouping, "step").agg(pl.col("utilization").mean()).sort("step")
    )

    grid = hv.Layout(
        dist.plot.heatmap(
            x="agent_pos_x",
            y="agent_pos_y",
            C="prob",
            row="agent",
            col="ratio_reward",
        )
        + dist.plot.heatmap(
            x="agent", y="ratio_reward", C="entropy", label="Entropy of Paths"
        )
        + data.plot.line(x="step", y="utilization", by=grouping, groupby="ratio_reward")
    ).cols(1)

    hvplot.show(grid, port=8080, responsive=True)

    hvplot.save(grid, "foo.html")

    # g = sns.FacetGrid(
    #     data,
    #     row="agent",
    #     col="name",
    #     hue="stage",
    #     sharex=True,
    #     sharey=True,
    #     margin_titles=True,
    # )
    # g.map_dataframe(
    #     sns.histplot,
    #     x="agent_pos_x",
    #     y="agent_pos_y",
    #     discrete=True,
    # ).savefig(FIGURE_PATH / "ratio_positions.png")
    #
    # sns.catplot(data=data, hue="agent", x="ratio", y="utilization", kind="boxen")
    #
    # sns.catplot(data=tile_agg, x="name", y="count", col="agent", hue="type", kind="bar")
    #
    # g = sns.FacetGrid(
    #     step_agg,
    #     col="name",
    #     hue="agent",
    #     sharex=True,
    #     sharey=True,
    #     margin_titles=True,
    # )
    # g.map_dataframe(sns.lineplot, x="step", y="rolling_reward")
    #
    # g = sns.FacetGrid(
    #     step_agg,
    #     col="name",
    #     hue="agent",
    #     sharex=True,
    #     sharey=True,
    #     margin_titles=True,
    # )
    # g.map_dataframe(sns.lineplot, x="step", y="cumulative_reward")
    #
    # g = sns.FacetGrid(
    #     step_agg,
    #     col="name",
    #     hue="agent",
    #     sharex=True,
    #     sharey=True,
    #     margin_titles=True,
    # )
    # g.map_dataframe(sns.lineplot, x="step", y="utilization")

    # multipage("all.pdf")


if __name__ == "__main__":
    main()
