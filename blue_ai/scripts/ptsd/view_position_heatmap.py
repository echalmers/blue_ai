import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path
import sys
from typing import List

from blue_ai.scripts.constants import DATA_PATH, N_TRIALS
from blue_ai.scripts.ptsd.train_agents import load_trial, save_results


def view_position_heatmap(directory: Path, agents_to_include: List[str], show_plots: bool, agent_state: str):
    """
    Generate heatmaps of agent positions across trials.

    This function aggregates agent trajectory data from multiple trials, extracts the (x, y) 
    positions, and creates heatmaps that visualize the frequency of positions occupied by 
    each agent. Heatmaps are normalized to a common scale across agents to enable fair 
    comparison of position distributions.

    Args:
        directory (Path): Path to the directory containing agent trial pickle files.
        agents_to_include (List[str]): List of agent class names to include in the analysis 
                                       (e.g., ["HealthyAgent", "PTSDAgent"]).
        show_plots (bool): Whether to display the generated plots.
        agent_state (str): Agent state identifier appended to trial filenames, used to 
    """

    # get the file names of the agents
    files = [
        f"{directory.name}/{agent}_{trial}{agent_state}.pkl"
        for trial in range(N_TRIALS)
        for agent in agents_to_include
    ]

    match agent_state:
        case "_traumatized_testing":
            title = 'Heatmap of agent positions after trauma'
            subpath = "position_heatmap_after_trauma.png"
        case "_exposure_therapy_testing":
            title = 'Heatmap of agent positions after exposure therapy'
            subpath = "position_heatmap_after_exposure_therapy.png"
        case "_exposure_therapy_2_testing":
            title = 'Heatmap of agent positions after exposure therapy 2'
            subpath = "position_heatmap_after_exposure_therapy_2.png"
        case "_relearned_testing":
            title = 'Heatmap of agent positions after relearning'
            subpath = "position_heatmap_after_relearning.png"
        case "_testing":
            title = 'Heatmap of agent positions before trauma'
            subpath = "position_heatmap_before_trauma.png"
        case "_connectivity_restoration_testing":
            title = 'Heatmap of agent positions after connectivity restoration'
            subpath = "position_heatmap_after_connectivity_restoration.png"
        case _:
            title = 'Heatmap of agent positions at unknown point'
            subpath = "position_heatmap_at_unknown_point.png"

    # concatinate all the results
    results = []
    for f in files:
        results.append(load_trial(DATA_PATH / f)[0])
    df = pd.concat(results, ignore_index=True)


    # split tuple into separate x, y columns
    df[["x", "y"]] = pd.DataFrame(df["position"].tolist(), index=df.index)

    # count frequency of each (x,y)
    heatmap_data = df.groupby(["agent","y", "x"]).size().unstack(fill_value=0)

    # reindex both y and x for each agent
    heatmap_data = (
        heatmap_data
        .reindex(columns=range(1, 7), fill_value=0)  # ensure 6 columns
        .reindex(pd.MultiIndex.from_product([heatmap_data.index.levels[0], range(1, 7)], 
                                            names=["agent","y"]), fill_value=0) # ensure 6 rows per agent
    )
    print(heatmap_data)

    # save the results
    results_path = directory / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    save_results(heatmap_data, results_path / subpath.replace("png", "pkl"))

    #plot heatmap
    img_path = directory / "img"
    img_path.mkdir(parents=True, exist_ok=True)

    num_agents = len(agents_to_include)
    _, axes = plt.subplots(1, num_agents, figsize=(4*num_agents, 4))

    # if only one agent, axes is not a list, so we wrap it
    if num_agents == 1:
        axes = [axes]

    # find global min and max for all agents
    global_min = heatmap_data.loc[agents_to_include].min().min()
    global_max = heatmap_data.loc[agents_to_include].max().max()

    for ax, agent in zip(axes, agents_to_include):
        data = heatmap_data.loc[agent]  # Select only rows for this agent
        sns.heatmap(data, annot=True, fmt="d", cmap="YlGnBu", ax=ax,
                    vmin=global_min, vmax=global_max)
        ax.set_title(agent)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.suptitle(title)
    plt.savefig(img_path/subpath)
    if show_plots:
        plt.show()



if __name__ == "__main__":
    agents_to_include : List[str] = [
        "HealthyAgent",
        "PTSDAgent",
        "TraumaSynapticDeficitAgent",
    ]
    view_position_heatmap(DATA_PATH / sys.argv[1], agents_to_include, show_plots=True, agent_state='_connectivity_restoration_testing')