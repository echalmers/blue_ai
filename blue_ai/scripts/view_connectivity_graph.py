# Paths to the files
import numpy as np
from matplotlib import pyplot as plt
import polars as pl

from blue_ai.scripts.constants import DATA_PATH

def shaded_slices(df):
    mean_data = np.mean(df, axis=0)
    std_data = np.std(df, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_data, label='Mean')
    plt.fill_between(mean_data - std_data, mean_data + std_data, color='gray', alpha=0.3, label='Std Dev')
    plt.title('Mean with Standard Deviation of Trials')
    plt.xlabel('Time')
    plt.ylabel('Measurement')
    plt.legend()
    plt.grid(True)
    plt.show()

# Define the paths to the files
# Read the parquet files and extract column '1' directly
sliced_healthy_1 = pl.read_parquet(DATA_PATH/"healthy_sliced_entropies.parquet")["1"]
sliced_depressed_1 = pl.read_parquet(DATA_PATH/"depressed_sliced_entropies.parquet")["1"]
sliced_treated_1 = pl.read_parquet(DATA_PATH/"treated_sliced_entropies.parquet")["1"]

stages = {}
# Concatenate the Series
sliced_entropies_layer_1 = pl.concat([sliced_healthy_1, sliced_depressed_1, sliced_treated_1])
print(len(sliced_entropies_layer_1))
plt.plot(sliced_entropies_layer_1, marker="o")
plt.axvline(x=len(sliced_healthy_1), color='r', linestyle='--', label='depressed')
plt.axvline(x=len(sliced_healthy_1)*2, color='green', linestyle='--', label='treated')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('entropies throughout 1 rehab trial')
plt.show()

healthy_entropies = pl.read_parquet(DATA_PATH/"healthy_entropies.parquet")["1"]
depressed_entropies = pl.read_parquet(DATA_PATH/"depressed_entropies.parquet")["1"]
treated_entropies = pl.read_parquet(DATA_PATH/"treated_entropies.parquet")["1"]
# Combine data for boxplot
full_rehab = [healthy_entropies, depressed_entropies, treated_entropies]

# Combine and box plot results
plt.boxplot(full_rehab)
plt.xticks([1, 2, 3], ['healthy', 'depressed', 'treated'])
plt.title('Connectivity entropy throughout the rehabilitation stages')
plt.show()


plt.plot(healthy_entropies, label="healthy", color='r')
plt.plot(depressed_entropies, label="depressed", color='b')
plt.plot(treated_entropies, label="treated", color='g')
plt.title("connectivity entropy for each trial")
plt.legend()
plt.show()