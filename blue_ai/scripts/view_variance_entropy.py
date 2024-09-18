from glob import glob
from scipy.stats import entropy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

def calculate_variance(neurons):
    # Use dictionary comprehension to compute variances for each layer
    variances = {
        layer: neurons.filter(pl.col('layer') == layer)['activation'].apply(lambda x: pl.Series(x).var())
        for layer in neurons['layer'].unique()
    }
    return variances

def calculate_distance_from_mean(variances):
    mean_variance = np.mean(variances)
    distance_from_mean = variances - mean_variance
    return distance_from_mean


def discretize_distances(distances, num_bins=10):
    hist, bin_edges = np.histogram(distances, bins=num_bins)
    # Normalize histogram to get probabilities
    probabilities = hist / np.sum(hist)
    return bin_edges, probabilities


def calculate_variance_entropy(filename, layers_of_interest=None, num_bins=10):
    var = calculate_variance(filename, layers_of_interest)
    dis_mean = calculate_distance_from_mean(var)
    _, discrete_distances = discretize_distances(dis_mean, num_bins)

    return entropy(discrete_distances, base=2)


def plot_variance_distributions(filename, layer_of_interest, agent_type):
    var = calculate_variance(filename, layer_of_interest)
    distances = calculate_distance_from_mean(var)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(distances, bins=10, edgecolor='k', alpha=0.7)
    plt.title(f'Normalized Variance Distribution for {agent_type} agent Layer {layer_of_interest}')
    plt.xlabel('Distance from Mean')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def plot_variance_over_time(filepath_1, layer_of_interest, agent_type_1, filepath_2=None, agent_type_2=None):
    var_1 = calculate_variance(filepath_1, layer_of_interest)
    distances_1 = calculate_distance_from_mean(var_1)
    fig, ax = plt.subplots(figsize=(10, 6))
    distances_1 = pd.Series(distances_1).rolling(1000).mean()
    ax.plot(distances_1, label=f'{agent_type_1}', color='blue')

    if filepath_2 is not None:
        var_2 = calculate_variance(filepath_2, layer_of_interest)
        distances_2 = calculate_distance_from_mean(var_2)
        distances_2 = pd.Series(distances_2).rolling(1000).mean()
        ax.plot(distances_2, label=f'{agent_type_2}', color='red')

    ax.set_title(f'Normalized Variance Over Time of Layer {layer_of_interest}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Distance from Mean')
    ax.legend()
    plt.show()


def main():
    # Define file paths
    rehab_healthy_files = sorted(
        glob(r'C:\Users\linda\PycharmProjects\blue_ai\blue_ai\scripts\data/rehabilitate_healthy_*_activations.pkl'))
    rehab_depressed_files = sorted(
        glob(r'C:\Users\linda\PycharmProjects\blue_ai\blue_ai\scripts\data/rehabilitate_depressed_*_activations.pkl'))
    rehab_treated_files = sorted(
        glob(r'C:\Users\linda\PycharmProjects\blue_ai\blue_ai\scripts\data/rehabilitate_treated_*_activations.pkl'))

    # Calculate entropies
    rehab_healthy_entropies = [calculate_variance_entropy(file, '1') for file in rehab_healthy_files]
    rehab_depressed_entropies = [calculate_variance_entropy(file, '1') for file in rehab_depressed_files]
    rehab_treated_entropies = [calculate_variance_entropy(file, '1') for file in rehab_treated_files]

    # Box Plot results
    full_rehab = [rehab_healthy_entropies, rehab_depressed_entropies, rehab_treated_entropies]
    plt.boxplot(full_rehab)
    plt.xticks([1, 2, 3], ['healthy', 'depressed', 'treated'])  # Set x-axis labels
    plt.title('Entropy throughout the rehabilitation stages')
    plt.show()

    #plot variance over time
    plot_variance_over_time(
        r'C:\Users\linda\PycharmProjects\blue_ai\blue_ai\scripts\data/rehabilitate_treated_0_activations.pkl',
        '1', "treated",
        r'C:\Users\linda\PycharmProjects\blue_ai\blue_ai\scripts\data/rehabilitate_healthy_9_activations.pkl',
        'healthy'
    )

    #plot entropy over time
    plt.plot(rehab_healthy_entropies, label='Entropy Over Time Healthy', color='blue', marker='o')
    plt.plot(rehab_treated_entropies, label='Entropy Over Time treated', color='red', linestyle='--', marker='x')
    plt.title("Variance entropy of healthy and treated")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
