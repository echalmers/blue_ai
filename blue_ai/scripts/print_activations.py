import pickle

import pandas as pd


def print_activations(data):
    #
    # # Iterate through activations
    # for layer_name, activation in activations.items():
    #     print(f"Layer: {layer_name}")


        print(f"Activation Tensor: {activation}")
        print(f"Shape: {activation.shape}")
        print(f"Mean: {activation.mean().item()}")
        print(f"Max: {activation.max().item()}")
        print(f"Min: {activation.min().item()}")
        print("=" * 40)

    # print(f"Activation Tensor: {data}")
    # print(f"Shape: {data.shape}")
    # print(f"Mean: {data.mean()}")
    # print(f"Max: {data.max()}")
    # print(f"Min: {data.min()}")
    # print("=" * 40)


# Load activations from file
with open(r'C:\Users\linda\PycharmProjects\blue_ai\blue_ai\scripts\data/rehabilitate_healthy_9_activations.pkl', "rb") as f:
    activations =
print_activations(activations)

