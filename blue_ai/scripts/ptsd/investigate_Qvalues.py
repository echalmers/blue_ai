import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sys
from pathlib import Path
from typing import List

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS
from blue_ai.scripts.ptsd.train_agents import load_trial



def investigate_Qvalues(directory: Path, agents_to_include: List[str], env: Image2VecWrapper, show_plots: bool, agent_state: str, single_plot: bool = True):
    """
    Investigate the Q-values of agents by loading them, evaluating them in a given 
    environment, and optionally plotting the results.

    This function loads agents from a specified directory, computes their Q-values for the 
    initial state of the environment, and aggregates the results into a DataFrame. If enabled, 
    it also generates comparative plots of the average Q-values across actions and agents.

    Args:
        directory (Path): Path to the directory containing the agent trial files.
        agents_to_include (List[str]): List of agent class names to include in the analysis 
            (e.g., ["HealthyAgent", "PTSDAgent"]).
        env (Image2VecWrapper): The wrapped environment used to evaluate the agents’ Q-values.
        show_plots (bool): Whether to display the Q-value plots after saving them.
        agent_state (str): Suffix indicating the state of the agents 
            (e.g., "_traumatized", "_exposure_therapy", "_relearned", or "").
        single_plot (bool, optional): Whether to generate a single consolidated plot comparing 
            agents’ Q-values. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - "agent": Name of the agent class.
            - "qvalues": Numpy array of Q-values for each available action.
    """

    # get the file names of the agents
    files = [
        f"{directory.name}/{agent}_{trial}{agent_state}.pkl"
        for trial in range(N_TRIALS)
        for agent in agents_to_include
    ]

    results = []
    for i in range(len(files)):
        filename = files[i]
        _, agent, _ = load_trial(DATA_PATH / filename)
        state, _ = env.reset()
        env.render()
        result = {
            "agent": agent.__class__.__name__,
            "qvalues": np.array(agent.get_action_values(state), dtype=np.float32)
        }
        results.append(result)
    
    results = pd.DataFrame(results)
    if single_plot:
        plotting(results, directory, show_plots, agent_state)

    return results
    

def plotting(df: pd.DataFrame, directory: Path, show_plots: bool, agent_state: str):

    folder_path = directory / "img"
    folder_path.mkdir(parents=True, exist_ok=True)

    # Expand Q_values into columns
    q_df = pd.DataFrame(df['qvalues'].tolist(), columns=['left', 'right', 'forward'])
    df = pd.concat([df[['agent']], q_df], axis=1)

    # Average per agent
    mean_df = df.groupby('agent').mean().reset_index()

    # Melt for seaborn
    melted_df = mean_df.melt(id_vars='agent', var_name='action', value_name='qvalue')

    # Plot as separate panels per agent
    sns.catplot(
        data=melted_df,
        x='action', y='qvalue',
        col='agent',
        kind='bar',
        sharey=True,  # same y-axis for easy comparison
        height=4, aspect=1
    )

    plt.subplots_adjust(top=0.85)

    match agent_state:
        case "_traumatized":
            title = "Average Q-values per Action after trauma, Split by Agent"
            subpath = "q_values_after_trauma.png"
        case "_exposure_therapy":
            title = "Average Q-values per Action after exposure therapy, Split by Agent"
            subpath = "q_values_after_exposure_therapy.png"
        case "_exposure_therapy_2":
            title = "Average Q-values per Action after exposure therapy 2, Split by Agent"
            subpath = "q_values_after_exposure_therapy_2.png"
        case "_relearned":
            title = "Average Q-values per Action after relearning, Split by Agent"
            subpath = "q_values_after_relearning.png"
        case "":
            title = "Average Q-values per Action before trauma, Split by Agent"
            subpath = "q_values_before_trauma.png"
        case _:
            title = "Average Q-values per Action at unspecified point, Split by Agent"
            subpath = "q_values_unspecified.png"

    plt.suptitle(title)
    plt.savefig(folder_path / subpath)
    if show_plots:
        plt.show()
    

if __name__ == "__main__":
    agents_to_include : List[str] = [
        "HealthyAgent",
        "PTSDAgent",
        "TraumaSynapticDeficitAgent",
    ]
    # place them in the exact position where they they would have gotten traumatized but without hazard
    env = Image2VecWrapper(
                TransientGoals(
                    render_mode="human", transient_reward=0.25, termination_reward=1, agent_start_pos=(4,4),
                    transient_locations=[[1,4],[4,2],[5,1]],
                    n_transient_obstacles=0,
                    #transient_obstacles=[[5,4]],
                    wall_locations =[[3,2],[3,3],[3,4],[3,5],[3,6]]
                )
            )
    investigate_Qvalues(DATA_PATH / sys.argv[1], agents_to_include, env, show_plots=True, agent_state='_traumatized')