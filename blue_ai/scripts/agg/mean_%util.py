from sys import argv
import polars as pl
import numpy as np
import seaborn as sns
from hvplot.polars import hvPlotTabularPolars as plot
import matplotlib.pyplot as plt
from tqdm import tqdm


## Columns use for group_by / subsetting
over = pl.selectors.matches("stage_[0-9]+_ratio") | pl.selectors.by_name(
    "agent",
    "stage",
    "ratio_reward",
    "trial_id",
)


def cumulative_over(x):
    return pl.col(x).cum_sum().over("agent", "trial_id", "stage")


def main():
    if not len(argv) > 1:
        print("Missing param 'filename' for data file")
        exit(1)

    data = (
        pl.scan_parquet(argv[1])
        .with_row_index()
        .with_columns(
            (pl.col("ratio_reward") / pl.col("ratio_penalty")).alias("ratio"),
            *[
                (
                    pl.col("ratios").list.get(n).cast(pl.String)
                    + ":"
                    + pl.col("ratios").list.get(n + 1).cast(pl.String)
                ).alias(f"stage_{n // 2}_ratio")
                for n in range(0, 4, 2)
            ],
        )
        .select(
            pl.all(),
            cumulative_reward=cumulative_over("reward"),
            util=(
                pl.col("reward")
                .cum_sum()
                .over("agent", "trial_id", "stage", "ratios")  #
                / pl.col("total_reward")
                .cum_sum()
                .over("agent", "trial_id", "stage", "ratios")
            ).rolling_mean(50),
        )
    )

    agg = (
        data.filter(pl.col("stage") == 1)
        .gather_every(200)
        .select(
            "util",
            "step",
            "agent",
            "stage_1_ratio",
            "stage_0_ratio",
        )
        .collect()
    )

    labels = (
        agg.select(pl.col("stage_0_ratio").unique().sort())
        .limit(8)["stage_0_ratio"]
        .to_list()
    )

    print("done agg")

    sns.relplot(
        data=agg,
        x="step",
        y="util",
        hue="agent",
        row="stage_1_ratio",
        col="stage_0_ratio",
        kind="line",
        facet_kws={"margin_titles": True},
    ).set_xticklabels(labels)
    plt.show()

    g = sns.FacetGrid(
        data.group_by(over)
        .agg(
            pl.col("ratio_reward").first(),
            pl.col("stage").first(),
            mean_util_per_step=pl.col("util").median(),
            std_util_per_step=pl.col("util").std(),
        )
        .collect()
        .sort("stage_0_ratio", "stage_1_ratio"),
        despine=True,
        margin_titles=True,
        row="stage_0_ratio",
        col="stage_1_ratio",
    )

    g.map_dataframe(
        sns.barplot,
        hue="agent",
        x="agent",
        y="mean_util_per_step",
    )

    plt.show()

    # accumulate.collect().sort(over).write_csv("./mean_%util.csv")


if __name__ == "__main__":
    main()
