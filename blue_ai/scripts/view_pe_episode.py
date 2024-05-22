from pandas import DataFrame
from blue_ai.scripts.train_agents import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt
from blue_ai.scripts.constants import FIGURE_PATH


def plot_learning_curves(
    ax,
    dataset: DataFrame,
):
    plt.sca(ax)
    plt.legend(title="Agent")

    print(dataset)

    # Group by agent and episode, calculate the max step and sum of rewards per episode
    grouped_dataset = (
        dataset.groupby(["agent", "step"])["reward"]
        .transform(lambda s: s.rolling(500))
        .mean()
    )

    print(grouped_dataset)

    sns.lineplot(
        data=grouped_dataset,
        y="reward",
        x="step",
        hue="agent",
        # levels=100,
        # fill=True,
        # alpha=(1 / 2),
    )


def main():
    # Load the dataset
    dataset = load_dataset("*.pkl")

    fig, ax = plt.subplots(figsize=(10, 6))

    plot_learning_curves(ax, dataset)

    # Show the plot
    plt.savefig(FIGURE_PATH / "reward_per_episode.png")
    plt.show()


if __name__ == "__main__":
    main()
