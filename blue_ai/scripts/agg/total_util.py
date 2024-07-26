import os
import polars as pl
from blue_ai.scripts.constants import DATA_PATH
import seaborn as sns
import matplotlib.pyplot as plt


def cumulative_over(x):
    return pl.col(x).cum_sum().over("agent", "trial_id", "stage", "ratio_reward")


pl.enable_string_cache()

stage_select = pl.selectors.matches("stage_[0-9]+_ratio")
data = (
    pl.scan_parquet(DATA_PATH / "full.parquet")
    .with_row_index()
    .with_columns(
        (cumulative_over("reward") / cumulative_over("total_reward")).alias("util"),
        cumulative_over("reward").alias("cum_rew"),
        *[
            (
                pl.col("ratios").list.get(n).cast(pl.String)
                + ":"
                + pl.col("ratios").list.get(n + 1).cast(pl.String)
            ).alias(f"stage_{n // 2}_ratio")
            for n in range(0, 4, 2)
        ],
    )
)


unique_column_values = (
    data.select(pl.col("agent").unique()).collect()["agent"].to_list()
)

over = pl.selectors.matches("stage_[0-9]+_ratio") | pl.col("trial_id")

if (DATA_PATH / "eric.cache.parquet").exists() and not "REDO" in os.environ:
    res = pl.read_parquet(DATA_PATH / "eric.cache.parquet")
else:
    res = (
        data.lazy()
        .filter(pl.col("stage") == 0)
        .group_by(over)
        .agg(
            (
                pl.col("reward").filter(pl.col("agent") == value).sum()
                / pl.col("total_reward").filter(pl.col("agent") == value).sum()
            ).alias(value)
            for value in unique_column_values
        )
        # .select(
        #     pl.selectors.matches("stage_[0-9]+_ratio"),
        #     (pl.col("HealthyAgent") - pl.col("RehabiliationAgent")).alias(
        #         "sum(Healthy Reward) - sum(Rehab Reward)"
        #     ),
        #     (pl.col("SpineLossDepression") - pl.col("RehabiliationAgent")).alias(
        #         "sum(SpineLossDepression) - sum(Rehab Reward)"
        #     ),
        # )
        .collect()
        .sort(pl.selectors.matches("stage_[0-9]+_ratio"))
        .melt(
            pl.selectors.matches("stage_[0-9]+_ratio")
            | pl.selectors.by_name("trial_id"),
            variable_name="agent",
            value_name="Total Utilization",
        )
    )

    res.write_parquet(DATA_PATH / "eric.cache.parquet")

res = res.with_columns(
    pl.concat_str(
        pl.selectors.matches("stage_[0-9]+_ratio"),
        separator="-",
        ignore_nulls=True,
    ).alias("Reward:Lava")
).drop(pl.selectors.matches("stage_[0-9]+_ratio"))


sns.barplot(
    data=res,
    x="Reward:Lava",
    y="Total Utilization",
    hue="agent",
)


plt.show()


print(res)
