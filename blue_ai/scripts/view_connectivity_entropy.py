import pickle
from glob import glob

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.stats import entropy
from tqdm.auto import tqdm

# test_dict = {
#     '1': np.array([
#         np.array([0.0, 0.5, 0.0, 0.1, 0.2]),
#         np.array([0.0, 0.5, 0.0, -0.1, 0.2]),
#         np.array([0.0, -0.5, 0.0, 0.1, 0.2]),
#         np.array([0.0, 0.5, 0.0, 0.1, -0.2]),
#         np.array([0.0, 0.5, 0.0, -0.1, -0.2]),
#         np.array([0.0, -0.5, 0.0, 0.1, 0.2]),
#         np.array([0.0, 0.5, 0.0, 0.1, -0.2]),
#         np.array([0.0, 0.5, 0.0, -0.1, -0.2])
#     ]),
#     '3': np.array([
#         np.array([0.01, -0.02, 0.03, 0.04, 0.05, 0.005, 0.3, 0.2, 0.4]),
#         np.array([-0.01, -0.02, 0.03, -0.04, 0.05, 0.005, 0.3, 0.2, 0.4]),
#         np.array([0.01, -0.02, -0.03, 0.04, 0.05, 0.005, 0.3, 0.2, 0.4]),
#         np.array([-0.01, 0.02, 0.03, 0.04, -0.05, -0.005, -0.3, -0.2, 0.4]),
#         np.array([0.01, -0.02, -0.03, 0.04, 0.05, 0.005, 0.3, 0.2, 0.4]),
#         np.array([-0.01, 0.02, 0.03, 0.04, -0.05, -0.005, -0.3, -0.2, 0.4]),
#         np.array([-0.01, -0.02, -0.03, -0.04, -0.05, 0.005, -0.3, 0.2, -0.4])
#     ])
# }


def build_ngrams(neurons, thresholds=None, n=2, split=False):


    #if no threshold is specified, we will use the mean of the layer
    if thresholds is None:
        thresholds = np.array([neurons_values.mean().item() for neurons_values in neurons.values()])

    layers_ngrams = {}
    for layer, neurons_values in neurons.items():
        threshold_i = 0
        active_layer_weights = []

        #filter out neurons that are less active than the threshold
        for time_step in tqdm(range(neurons_values.shape[0]), position=0, leave=True, desc="processing time steps"):
            active_weights_now = np.where(neurons_values[time_step] >= thresholds[threshold_i])[0]
            active_layer_weights.append(active_weights_now)
        threshold_i += 1

        #setup for building ngram-model
        ngrams = {}
        ngrams_amount = 0

        #count ngrams
        for i in tqdm(range(len(active_layer_weights) - n + 1), position=0, leave=True, desc="counting ngrams"):
            ngram = tuple(map(tuple, active_layer_weights[i:i + n]))
            if ngram in ngrams:
                ngrams[ngram] += 1
            else:
                ngrams[ngram] = 1
            ngrams_amount += 1

        layers_ngrams[layer] = ngrams

        #transform ngram-count into probabilities
        for ngram in ngrams:
            ngrams[ngram] /= ngrams_amount

    return layers_ngrams


def calculate_ngram_entropies(filepaths, layer='1'):
    entropies = []
    for file in filepaths:
        connectivity = build_ngrams(file, thresholds=None, n=2)
        entropy_value = entropy(list(connectivity[layer].values()), base=2)
        entropies.append(entropy_value)
    return entropies




def main():

    # Paths to the files
    rehab_healthy = sorted(
        glob(r'C:\Users\linda\PycharmProjects\blue_ai\blue_ai\scripts\data/rehabilitate_healthy_*_activations.pkl'))
    rehab_depressed = sorted(
        glob(
            r'C:\Users\linda\PycharmProjects\blue_ai\blue_ai\scripts\data/rehabilitate_depressed_*_activations.pkl'))
    rehab_treated = sorted(
        glob(r'C:\Users\linda\PycharmProjects\blue_ai\blue_ai\scripts\data/rehabilitate_treated_*_activations.pkl'))

    # Calculate entropies for each category
    rehab_healthy_entropies = calculate_ngram_entropies(rehab_healthy)
    rehab_depressed_entropies = calculate_ngram_entropies(rehab_depressed)
    rehab_treated_entropies = calculate_ngram_entropies(rehab_treated)


    # Combine and box plot results
    full_rehab = [rehab_healthy_entropies, rehab_depressed_entropies, rehab_treated_entropies]
    plt.boxplot(full_rehab)
    plt.xticks([1, 2, 3], ['healthy', 'depressed', 'treated'])
    plt.title('Connectivity entropy throughout the rehabilitation stages')
    plt.show()

    #plot entropy over time
    plt.plot(rehab_healthy_entropies, label='Entropy Over Time Healthy', color='blue', marker='o')
    plt.plot(rehab_treated_entropies, label='Entropy Over Time treated', color='red', linestyle='--', marker='x')
    plt.title("connectivity entropy of healthy and treated")
    plt.legend()
    plt.show()


    with open(r'C:\Users\linda\PycharmProjects\blue_ai\blue_ai\scripts\data/rehabilitate_treated_0_activations.pkl', "rb") as f:
        treated_0 = pickle.load(f)

    batch_num = int(10_000/500)
    # Create a list to store the smaller dictionaries
    split_dicts = [{} for _ in range(batch_num)]
    split_entropies = []

    # split activations
    for layer_name, activations in treated_0.items():
        split_tensors = torch.split(activations, batch_num, dim=0)

        # Assign each chunk to a new dictionary
        for i in range(batch_num):
            split_dicts[i][layer_name] = split_tensors[i]
            connectivity = build_ngrams(split_dicts[i], n=2)
            ent = entropy(list(connectivity['1'].values()), base=2)
            split_entropies.append(ent)
            print(f'entropy{i} ', ent)

    print(split_entropies)
    #split_normalized = pd.Series(split_entropies).rolling(10).mean()
    plt.plot(split_entropies)
    plt.xlabel("batches")
    plt.ylabel("entropy")
    plt.title(f"connectivity entropy every {10_000/batch_num} steps")
    plt.show()

if __name__ == '__main__':
    main()
