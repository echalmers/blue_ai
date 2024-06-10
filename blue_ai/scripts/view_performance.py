import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from blue_ai.scripts.train_agents import load_dataset
from blue_ai.envs.transient_goals import TransientGoals
import blue_ai.agents.agent_classes as agent_classes

from blue_ai.scripts.constants import FIGURE_PATH


def aggregate_goals(type, data, include_lava=True):
    if type == "total":
        goals = (
            data.groupby(by=["trial_id", "agent"])[
                ["terminal_goal", "transient_goal", "lava", "stuck"]
            ]
            .sum()
            .reset_index()
        )
    elif type == "episode":
        goals = (
            data.groupby(by=["trial_id", "episode", "agent"])[
                ["terminal_goal", "transient_goal", "lava", "stuck"]
            ]
            .sum()
            .reset_index()
        )
        goals = (
            goals.groupby(by=["trial_id", "agent"])[
                ["terminal_goal", "transient_goal", "lava", "stuck"]
            ]
            .mean()
            .reset_index()
        )

    goals_transient = goals[["trial_id", "agent", "transient_goal"]]
    goals_transient["event"] = "optional goal"
    goals_transient.rename({"transient_goal": "count"}, axis=1, inplace=True)

    goals_terminal = goals[["trial_id", "agent", "terminal_goal"]]
    goals_terminal["event"] = "required goal"
    goals_terminal.rename({"terminal_goal": "count"}, axis=1, inplace=True)

    lava = goals[["trial_id", "agent", "lava"]]
    lava["event"] = "hazard"
    lava.rename({"lava": "count"}, axis=1, inplace=True)

    goals = pd.concat(
        [goals_terminal, goals_transient] + ([lava] if include_lava else []),
        ignore_index=True,
    )
    goals.rename({"event": "object"}, axis=1, inplace=True)
    return goals


class PerformancePlotter:

    def __init__(
        self,
        agent_classes=(
            agent_classes.HealthyAgent,
            agent_classes.SpineLossDepression,
        ),
    ):

        self.agent_classes = agent_classes
        self.high_terminal_results = load_dataset(
            [f"{cls.__name__}_[!s]*.pkl" for cls in agent_classes]
        )

        print(self.high_terminal_results)

    @staticmethod
    def plot_sample_env(ax):
        plt.sca(ax)

        # plot sample env
        env = TransientGoals(
            img_filename="env1.png",
            transient_locations=[(4, 1), (5, 4)],
            transient_obstacles=[(2, 5)],
            agent_start_pos=(2, 3),
            agent_start_dir=1,
            render_mode="rgb_array",
        )
        env.reset()

        plt.imshow(env.render())
        plt.xticks([])
        plt.yticks([])

    def plot_learning_curves(self, ax, n_boot=1, **kwargs):
        plt.sca(ax)

        # high_terminal_results['avg_reward'] = high_terminal_results.groupby(['trial_id', 'agent'])['reward'].transform(lambda x: x.rolling(250).mean())

        # plot cumulative reward
        sns.lineplot(
            data=self.high_terminal_results[
                # (self.high_terminal_results["step"] <= 20_000)
                (self.high_terminal_results["step"] % 5 == 0)
            ],
            x="step",
            y="cumulative_reward",
            hue="agent",
            n_boot=n_boot,
            **kwargs,
        )
        plt.title("cumulative reward obtained")
        plt.ylabel("")
        plt.xlabel("time (steps in environment)")

    def plot_goals_per_episode(self, ax, n_boot=1):
        plt.sca(ax)

        high_terminal_goals = aggregate_goals(
            type="episode", data=self.high_terminal_results
        )
        sns.barplot(
            data=high_terminal_goals,
            x="agent",
            y="count",
            hue="object",
            n_boot=n_boot,
            palette=["tab:green", "tab:blue", "tab:red"],
            order=[a.display_name for a in self.agent_classes],
        )
        plt.title("objects reached per episode")
        plt.ylabel("")
        # sns.move_legend(plt.gca(), "upper left")
        plt.xlabel("type of goal")
        plt.xlabel("")


if __name__ == "__main__":
    plotter = PerformancePlotter()

    f, ax = plt.subplots(1, 2, figsize=(9, 3))
    # plot_sample_env(ax[0])

    plt.subplot(1, 2, 1)
    plotter.plot_learning_curves(ax[0])

    plt.subplot(1, 2, 2)
    plotter.plot_goals_per_episode(ax[1])

    plt.show()
    plt.savefig(FIGURE_PATH / "performance.png")
    exit()
