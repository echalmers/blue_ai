from glob import glob
from scipy.stats import entropy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from blue_ai.scripts.constants import DATA_PATH


def calculate_variance(neurons):

    exploded_activations = neurons.explode("activation")
    variances = exploded_activations.group_by(["step", "layer"]).agg([
    pl.col("activation").var().alias("activation_variance")
    ])
    return variances


def calculate_distance_from_mean(variances):
    layer_means = variances.group_by("layer").agg([pl.col("activation_variance").mean().alias("mean_variance")])
    variances = variances.join(layer_means, on="layer")
    distances = variances.with_columns([
        (pl.col("activation_variance") - pl.col("mean_variance")).alias("distance_from_mean")
    ])
    return distances


def discretize_distances(distances, num_bins=10):
    layer_hist = {}
    layers = distances.select("layer").unique().to_series()

    for layer in layers:
        layer_distances = distances.filter(pl.col("layer") == layer)
        distance_values = np.array(layer_distances["distance_from_mean"].to_list())
        hist, bin_edges = np.histogram(distance_values, bins=num_bins)
        # Normalize histogram to get probabilities
        probabilities = hist / np.sum(hist)
        layer_hist[layer[0]] = {
            "bin_edges": bin_edges,
            "probabilities": probabilities
        }

    return layer_hist


def calculate_variance_entropy(data, layer, num_bins=10):
    var = calculate_variance(data)
    dis_mean = calculate_distance_from_mean(var)
    histograms = discretize_distances(dis_mean, num_bins)

    return entropy(histograms[layer]["probabilities"], base=2)


def plot_variance_distributions(data, layer_of_interest, agent_type):
    var = calculate_variance(data)
    distances = calculate_distance_from_mean(var)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(distances["distance_from_mean"], bins=10, edgecolor='k', alpha=0.7)
    plt.title(f'Normalized Variance Distribution for {agent_type} agent Layer {layer_of_interest}')
    plt.xlabel('Distance from Mean')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def plot_variance_over_time(data, layer_of_interest, agent_type):
    var = calculate_variance(data)
    distances = calculate_distance_from_mean(var)
    distances = pd.Series(distances["distance_from_mean"]).rolling(7000).mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(distances, label=f'{agent_type}', color='blue')
    ax.set_title(f'Normalized Variance Over Time of Layer {layer_of_interest}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Distance from Mean')
    ax.legend()
    plt.show()


def main():
    stages = ["healthy", "depressed", "treated"]
    all_entropy = []
    for stage in stages:
        stage_entropy = []
        for i in range(9):
            filename = DATA_PATH / f'rehabilitate_{stage}_{i}_activations.parquet'
            datafile = pl.read_parquet(filename)
            stage_entropy.append(calculate_variance_entropy(datafile, "1"))
        all_entropy.append(stage_entropy)

    data = pl.read_parquet(DATA_PATH/"rehabilitate_treated_7_activations.parquet")
    # plot_variance_distributions(data, "1", "depressed")
    plot_variance_over_time(data, "1", "depressed")



if __name__ == "__main__":
    main()
