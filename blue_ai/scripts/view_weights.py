import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import imageio
from tqdm import tqdm
from blue_ai.scripts.constants import DATA_PATH


def generate_heatmap(index, df, input_size, output_size, layer):

    # Extract the weights for the given step
    step_weights = df.filter(pl.col("index") == index)[layer].to_list()[0]
    weights_array = np.array(step_weights).reshape(input_size, output_size)
    stage = df.filter(pl.col("index") == index)["stage"].to_list()[0]
    #generate heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(weights_array, cmap="coolwarm", cbar=True, ax=ax)

    step = df.filter(pl.col("index") == index)["step"].to_list()[0]
    ax.set_title(f"Weights Heatmap at step {step} - {stage.capitalize()} Stage")
    ax.set_xlim(0, output_size)
    ax.set_ylim(0, input_size)

    #convert heatmap into an image
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)  # Close the figure after saving the image
    return image


def main():
    id = "3"
    all_dfs = []
    stages = ["healthy", "depressed", "entropic", "treated"]

    for stage in stages:
        filename = DATA_PATH / f'{id}_{stage}_1_weights.parquet'
        datafile = pl.read_parquet(filename)
        datafile = datafile.with_columns([pl.lit(stage).alias("stage")])
        # datafile["stage"] = stage
        all_dfs.append(datafile)
        print(f"{stage}: {len(datafile)}")

    merged_weights = pl.concat(all_dfs, how="vertical")
    merged_weights = merged_weights.with_row_index("index")

    input_size = 100
    output_size = 10

    # Get the unique steps
    steps = merged_weights["index"].to_list()
    images = []
    for step in tqdm(steps, desc="Generating Heatmaps", unit="step", leave=True, dynamic_ncols=True):
        image = generate_heatmap(step, merged_weights, input_size, output_size, "1.weights")
        images.append(image)

    # Save the images as a GIF
    gif_path = DATA_PATH / f"test_{id}_weights_heatmap.gif"
    imageio.mimsave(gif_path, images, fps=10)

    print(f"GIF saved at {gif_path}")


if __name__ == '__main__':
    main()
