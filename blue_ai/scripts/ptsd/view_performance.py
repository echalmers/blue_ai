import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch import nn

from pathlib import Path
import sys
from typing import List

from blue_ai.agents.agent_classes import BaseAgent, HealthyAgent, PTSDAgent, TraumaSynapticDeficitAgent
import blue_ai.agents.agent_classes as classes
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.scripts.constants import DATA_PATH
from blue_ai.scripts.ptsd.train_agents import load_dataset


class PerformancePlotter:

    def __init__(
        self,
        directory: Path,
        agent_state: str,
        agent_classes=None,
        results_dataframe=None
    ):

        if agent_classes is None and results_dataframe is None:
            agent_classes = (
                classes.HealthyAgent,
                classes.PTSDAgent,
                classes.TraumaSynapticDeficitAgent
            )

        if results_dataframe is not None:
            self.high_terminal_results = results_dataframe
            self.agent_classes = None

        else:
            self.agent_classes = agent_classes
            self.high_terminal_results = load_dataset(
                [f"{directory.name}/{cls.__class__.__name__}_[!s]{agent_state}.pkl" for cls in agent_classes]
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

    @staticmethod
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

    def plot_goals_per_episode(self, ax, n_boot=1, last_n_steps=None):
        plt.sca(ax)

        high_terminal_goals = self.aggregate_goals(
            type="episode",
            data=self.high_terminal_results.groupby(['trial_id', 'agent']).tail(last_n_steps or self.high_terminal_results.shape[0])
        )
        sns.barplot(
            data=high_terminal_goals,
            x="agent",
            y="count",
            hue="object",
            n_boot=n_boot,
            palette=["tab:green", "tab:blue", "tab:red"],
            order=[a.display_name for a in self.agent_classes] if self.agent_classes else None,
            errorbar=('pi', 95)
        )
        plt.title("objects reached per episode")
        plt.ylabel("")
        # sns.move_legend(plt.gca(), "upper left")
        plt.xlabel("type of goal")
        plt.xlabel("")


def view_performance(directory: Path, agents_to_include: List[str], name_suffix: str, show_plots: bool, agent_state: str):
    """
    Generate and save performance plots for a set of agents based on trial results.

    This function uses the PerformancePlotter class to visualize agent performance in terms of:
    1. Cumulative reward over time during training/testing.
    2. Number of goals (required, optional, hazards) reached per episode.

    The plots are saved as an image file in a subdirectory "img" within the provided directory.
    Optionally, the plots can also be displayed interactively.

    Args:
        directory (Path): Path to the directory containing trial results and where plots will be saved.
        agents_to_include (List[str]): List of agent class instances to include in the plots.
        name_suffix (str): Suffix to append to the saved plot filename for identification.
        show_plots (bool): If True, the plots will be displayed interactively.
        agent_state (str): Identifier string for the state of the agents (e.g., "_traumatized", "_relearned").
    """

    plotter = PerformancePlotter(directory, agent_state, agent_classes= agents_to_include)

    folder_path = directory / "img"
    folder_path.mkdir(parents=True, exist_ok=True)

    f, ax = plt.subplots(1, 2, figsize=(12, 4))
    f.suptitle(name_suffix)
    plt.subplot(1, 2, 1)
    plotter.plot_learning_curves(ax[0])

    plt.subplot(1, 2, 2)
    plotter.plot_goals_per_episode(ax[1])
    
    plt.savefig(folder_path/f"performance{name_suffix}.png")
    if show_plots:
        plt.show()


if __name__ == "__main__":
    network = nn.Sequential(
        nn.Flatten(1, -1),
        nn.Linear(100, 25), nn.Tanh(),
        nn.Linear(25, 4)
    )
    
    agents: List[BaseAgent] = [
        HealthyAgent(network= network),
        PTSDAgent(network= network),
        TraumaSynapticDeficitAgent(network= network)
    ]

    view_performance(DATA_PATH / sys.argv[1], agents, "_testing_in_learning_env_after_new_therapy", True, '_relearned')
