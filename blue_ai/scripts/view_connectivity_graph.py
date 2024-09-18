# Paths to the files
import numpy as np
from matplotlib import pyplot as plt
import polars as pl
import seaborn as sns

from blue_ai.scripts.constants import DATA_PATH


def main():
    stages = ["healthy", "depressed", "treated"]
    all_trials_data = []

    for i in range(9):
        trial_data = pl.DataFrame()  # Initialize an empty DataFrame for each trial
        stages_length = []
        for stage in stages:
            stage_data = pl.read_parquet(DATA_PATH / f"{stage}_{i}_sliced_entropies.parquet")[["1"]]
            num_rows = stage_data.shape[0]
            stages_length.append(num_rows)

            if trial_data.is_empty():
                trial_data = stage_data
            else:
                trial_data = trial_data.vstack(stage_data)

        # Generate an index column (time points) using pl.int_range() and pl.len()
        trial_data = trial_data.select(
            pl.int_range(0, pl.len(), dtype=pl.UInt32).alias("time point"),
            pl.all()
        ).with_columns([
            pl.lit(i).alias("trial")
        ])

        all_trials_data.append(trial_data)
        cumulative_time_points = [sum(stages_length[:j + 1]) for j in range(len(stages_length))]

    combined_data = pl.concat(all_trials_data)
    combined_data = combined_data.to_pandas()

    sns.lineplot(data=combined_data, x="time point", y="1", errorbar="sd", estimator="mean")

    for i, time_point in enumerate(cumulative_time_points[:-1]):
        plt.axvline(x=time_point, color='red', linestyle='--')

    plt.title("connectivity entropy over the trials")
    plt.xlabel("Time Point")
    plt.ylabel("Entropy")
    plt.legend()
    plt.show()


    # # Define the paths to the files
    # # Read the parquet files and extract column '1' directly
    # sliced_healthy_1 = pl.read_parquet(DATA_PATH / "healthy_0_sliced_entropies.parquet")["1"]
    # sliced_depressed_1 = pl.read_parquet(DATA_PATH / "depressed_0_sliced_entropies.parquet")["1"]
    # sliced_treated_1 = pl.read_parquet(DATA_PATH / "treated_0_sliced_entropies.parquet")["1"]
    #
    # healthy_entropies = pl.read_parquet(DATA_PATH / "healthy_entropies.parquet")["1"]
    # depressed_entropies = pl.read_parquet(DATA_PATH / "depressed_entropies.parquet")["1"]
    # treated_entropies = pl.read_parquet(DATA_PATH / "treated_entropies.parquet")["1"]
    #
    # # Combine data for boxplot
    # full_rehab = [healthy_entropies, depressed_entropies, treated_entropies]


if __name__ == '__main__':
    main()
