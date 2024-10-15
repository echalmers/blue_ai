import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import imageio
from tqdm import tqdm
from blue_ai.scripts.constants import DATA_PATH


# Function to generate a heatmap and return it as an image (for each step)
def generate_heatmap(step, df, input_size, output_size, layer):

    # Extract the weights for the given step
    step_weights = df.filter(pl.col("step") == step)[layer].to_list()[0]
    weights_array = np.array(step_weights).reshape(input_size, output_size)
    stage = df.filter(pl.col("step") == step)["stage"].to_list()[0]

    #generate heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(weights_array, cmap="coolwarm", cbar=True, ax=ax)

    ax.set_title(f"Weights Heatmap at step {step} - {stage.capitalize()} Stage")
    ax.set_xlim(0, output_size)
    ax.set_ylim(0, input_size)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)  # Close the figure after saving the image
    return image


def main():
    id = "4"
    all_dfs = []
    stages = ["healthy", "depressed", "entropic", "treated"]

    for stage in stages:
        filename = DATA_PATH / f'4_{stage}_0_0_weights.parquet'
        datafile = pl.read_parquet(filename)
        datafile = datafile.with_columns([pl.lit(stage).alias("stage")])
        all_dfs.append(datafile)
        print(stage)

    merged_weights = pl.concat(all_dfs, how="vertical")
    merged_weights = merged_weights.with_row_count()
    breakpoint()
    # Define input and output size
    input_size = 100
    output_size = 10


    images = []
    for step in tqdm(steps, desc="Generating Heatmaps", unit="step", leave=True):
        image = generate_heatmap(step, merged_weights, input_size, output_size, "1.weights")
        images.append(image)

    # Save the images as a GIF
    gif_path = DATA_PATH / f"{id}_weights_heatmap.gif"
    imageio.mimsave(gif_path, images, fps=10)

    print(f"GIF saved at {gif_path}")


if __name__ == '__main__':
    main()
