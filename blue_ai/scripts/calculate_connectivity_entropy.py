import polars as pl
from scipy.stats import entropy
from tqdm.auto import tqdm

from blue_ai.scripts.constants import DATA_PATH


def build_ngrams(neurons, thresholds=None, n=2, split=False):
    layers_ngrams = {}

    #if no threshold is specified, we will use the mean of the layer
    if thresholds is None:
        thresholds = {
            layer: neurons.filter(pl.col('layer') == layer)['activation'].explode().mean()
            for layer in neurons['layer'].unique()
        }

    for layer in neurons['layer'].unique():
        layer_neurons = neurons.filter(pl.col('layer') == layer)
        activations = layer_neurons['activation']
        threshold = thresholds[layer]

        active_layer = []
        #filter out neurons that are less active than the threshold
        for time_step in tqdm(activations, desc=f"Processing layer {layer} activations"):
            active_neurons = [neuron_index for neuron_index, act in enumerate(time_step) if act >= threshold]
            active_layer.append(active_neurons)

        #setup for building ngram-model
        ngrams = {}
        ngrams_amount = 0

        #count ngrams
        for i in tqdm(range(len(active_layer) - n + 1), position=0, leave=True, desc="counting ngrams"):
            ngram = tuple(tuple(active_layer[j]) for j in range(i, i + n))

            if ngram in ngrams:
                ngrams[ngram] += 1
            else:
                ngrams[ngram] = 1
            ngrams_amount += 1

        #transform ngram-count into probabilities
        for ngram in ngrams:
            ngrams[ngram] /= ngrams_amount

        layers_ngrams[layer] = ngrams

    return layers_ngrams


def calculate_ngram_entropy(df):
    # Build n-grams
    connectivity = build_ngrams(df, thresholds=None, n=2)
    return {layer: entropy(list(connectivity[layer].values())) for layer in connectivity}


def calculate_sliced_entropies(df, slices):
    entropies = {}
    for layer in df['layer'].unique():
        layer_neurons = df.filter(pl.col('layer') == layer)
        entropies[layer] = []
        rows_per_slice = len(layer_neurons) // slices
        for slice in range(slices):
            start_index = slice * rows_per_slice
            end_index = None if slice == slices - 1 else (slice + 1) * rows_per_slice
            layer_slice = layer_neurons.slice(start_index, end_index - start_index if end_index else None)
            sliced_entropy = (calculate_ngram_entropy(layer_slice))
            entropy_list = list(sliced_entropy.values())
            entropies[layer].append(entropy_list[0])
    return entropies


def main():
    stages = ["healthy", "depressed", "entropic", "treated"]
    id = "3"
    slice_sizes = []
    for i in range(0, 5):
        for stage in stages:
            filename = DATA_PATH / f'{id}_{stage}_{i}_activations.parquet'
            stage_data = pl.read_parquet(filename)
            layer_amount = len(stage_data['layer'].unique())
            slices = len(stage_data)//(4_500 * layer_amount)
            print(f"{stage}: {len(stage_data)}, {slices} slices")
            slice_sizes.append(slices)
            sliced_entropies = pl.DataFrame(calculate_sliced_entropies(stage_data, slices=slices))
            sliced_entropies.write_parquet(DATA_PATH / f"{id}_{stage}_{i}_sliced_entropies.parquet")

    print(slice_sizes)
if __name__ == '__main__':
    main()
