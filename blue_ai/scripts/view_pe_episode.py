from blue_ai.scripts.train_agents import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt
from blue_ai.scripts.constants import FIGURE_PATH


def plot_learning_curves(ax, dataset, n_boot=1, **kwargs):
    plt.sca(ax)

    # Group by agent and episode, calculate the max step and sum of rewards per episode
    grouped_dataset = (
        dataset.groupby(["agent", "episode"])
        .agg({"step": "max", "reward": "sum"})
        .reset_index()
    )

    # Calculate the moving average of the reward for each agent
    grouped_dataset["moving_avg_reward"] = grouped_dataset.groupby("agent")[
        "reward"
    ].transform(lambda x: x.rolling(window=100, min_periods=1).mean())

    # Plot the moving average reward per episode
    sns.lineplot(
        data=grouped_dataset,
        x="step",
        y="moving_avg_reward",
        hue="agent",
        n_boot=n_boot,
        **kwargs,
    )

    plt.title("Moving Average Reward per Episode by Agent")
    plt.ylabel("Moving Average Reward")
    plt.xlabel("Step")
    plt.legend(title="Agent")


def main():
    # Load the dataset
    dataset = load_dataset("*.pkl")

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot learning curves
    plot_learning_curves(ax, dataset)

    # Show the plot
    plt.savefig(FIGURE_PATH / "reward_per_episode.png")
    plt.show()


if __name__ == "__main__":
    main()
