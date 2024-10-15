# Paths to the files
import numpy as np
from matplotlib import pyplot as plt
import polars as pl
import seaborn as sns

from blue_ai.scripts.constants import DATA_PATH


def main():
    id = "4"
    stages = ["healthy", "depressed", "treated"]
    all_trials_data = []

    for i in range(9):
        trial_data = pl.DataFrame()
        stages_length = []
        for stage in stages:
            stage_data = pl.read_parquet(DATA_PATH / f"{id}_{stage}_{i}_sliced_entropies.parquet")
            breakpoint()
            # num_rows = stage_data.shape[0]
            # stages_length.append(num_rows)

            if trial_data.is_empty():
                trial_data = stage_data
            else:
                trial_data = trial_data.vstack(stage_data)

        trial_data = trial_data.select(
            pl.int_range(0, pl.len(), dtype=pl.UInt32).alias("time point"),
            pl.all()
        ).with_columns([
            pl.lit(i).alias("trial")
        ])

        all_trials_data.append(trial_data)
        # cumulative_time_points = [sum(stages_length[:j+1]) for j in range(len(stages_length))]

    combined_data = pl.concat(all_trials_data)

    sns.lineplot(data=combined_data, x="time point", y="1", errorbar="sd", estimator="mean")

    # for i, time_point in enumerate(cumulative_time_points[:-1]):
    #     plt.axvline(x=time_point, color='red', linestyle='--')

    plt.title("connectivity entropy long depression")
    plt.xlabel("Time Point")
    plt.ylabel("Entropy")
    plt.legend()
    plt.show()


    # full_rehab = pl.DataFrame()
    # for stage in stages:
    #     stage_data = pl.read_parquet(DATA_PATH / f"{stage}_dropconnect_sliced_entropies.parquet")[["fc1"]]
    #
    #     if full_rehab.is_empty():
    #         full_rehab = stage_data
    #     else:
    #         full_rehab = full_rehab.vstack(stage_data)
    #
    #
    # full_rehab = full_rehab.select(
    #     pl.int_range(0, pl.len(), dtype=pl.UInt32).alias("time_point"),
    #     pl.all()
    # )
    # sns.lineplot(full_rehab, x="time_point", y="fc1")
    # plt.show()

if __name__ == '__main__':
    main()
