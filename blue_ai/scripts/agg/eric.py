import polars as pl
from polars.dataframe import DataFrame


def cumulative_over(x):
    return pl.col(x).cum_sum().over("agent", "trial_id", "stage", "ratio_reward")


pl.enable_string_cache()

stage_select = pl.selectors.matches("stage_[0-9]+_ratio")
data = pl.scan_parquet("~/Downloads/ratios_layered.parquet").with_columns(
    (cumulative_over("reward") / cumulative_over("total_reward")).alias("util"),
    *[
        (
            pl.col("ratios").list.get(n).cast(pl.String)
            + ":"
            + pl.col("ratios").list.get(n + 1).cast(pl.String)
        ).alias(f"stage_{n // 2}_ratio")
        for n in range(0, 4, 2)
    ],
)


def mapper(group: DataFrame):
    healthy = pl.col("HealthyAgent")
    rehab = pl.col("RehabiliationAgent")
    dep = pl.col("SpineLossDepression")

    x = (
        group.filter(pl.col("util").is_finite() & pl.col("util").is_not_null())
        .pivot(
            values="util",
            index=["stage_0_ratio", "step"],
            columns="agent",
            aggregate_function="sum",
        )
        .select((healthy - rehab), (dep - rehab))
    )

    return x


res = (
    data.filter(pl.col("stage") == 0)
    .group_by("stage_0_ratio")
    .map_groups(mapper, schema=None)
    .collect()
)

print(res)
