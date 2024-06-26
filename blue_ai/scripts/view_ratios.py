import polars as pl
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from blue_ai.scripts.constants import DATA_PATH, FIGURE_PATH


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format="pdf")
    pp.close()


def main():
    disk_data = pl.scan_parquet(DATA_PATH / "ratios.parquet")

    data = disk_data.select(~pl.selectors.by_name("params")).collect()

    numeric_lists = pl.selectors.by_dtype([pl.List(T) for T in pl.NUMERIC_DTYPES])
    unique = ["agent", "trial_id", "name"]

    data = data.with_columns(
        data.group_by(unique, maintain_order=True)
        .agg(
            pl.col("reward").alias("cumulative_reward").cum_sum(),
            (pl.col("reward").cum_sum() / pl.col("total_reward").cum_sum())
            .rolling_mean(20)
            .alias("utilization"),
            (pl.col("ratio_reward") / pl.col("ratio_penalty")).alias("ratio"),
            pl.col("reward").rolling_mean(2000).name.prefix("rolling_"),
        )
        .explode(numeric_lists)
    )

    step_agg = data.group_by("agent", "name", "step").agg(pl.selectors.numeric().mean())

    g = sns.FacetGrid(
        data,
        row="agent",
        col="name",
        sharex=True,
        sharey=True,
        margin_titles=True,
    )
    g.map_dataframe(
        sns.histplot,
        x="agent_pos_x",
        y="agent_pos_y",
        discrete=True,
    ).savefig(FIGURE_PATH / "ratio_positions.png")

    fig = plt.figure(0, ((6 * 4), 6))
    ax = fig.subplots(1, 3)

    ax = ax.flatten()
    sns.lineplot(
        ax=ax[0],
        data=step_agg,
        style="agent",
        hue="name",
        x="step",
        y="cumulative_reward",
    )

    sns.lineplot(
        ax=ax[1],
        data=step_agg,
        hue="agent",
        x="step",
        y="spine_loss",
    )

    sns.lineplot(
        ax=ax[2],
        data=data,
        hue="agent",
        x="ratio",
        y="utilization",
    )

    g = sns.FacetGrid(
        step_agg,
        col="name",
        sharex=True,
        sharey=True,
        margin_titles=True,
        legend_out=True,
    )
    g.map_dataframe(
        sns.lineplot,
        hue="agent",
        x="step",
        y="rolling_reward",
    )

    g.add_legend()

    multipage("all.pdf")


if __name__ == "__main__":
    main()
