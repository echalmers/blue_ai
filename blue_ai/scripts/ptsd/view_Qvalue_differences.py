import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
from pathlib import Path
from typing import List

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH
from blue_ai.scripts.ptsd.investigate_Qvalues import investigate_Qvalues

def view_Qvalue_differences(directory: Path, agents_to_include: List[str], agent_state: str, env: Image2VecWrapper):
    """
    Compare agents' Q-values at the trauma location versus other positions in the environment.

    This function evaluates multiple agents at all positions in the environment, 
    excluding walls, transient goals, and the terminal position. For each position, it 
    computes the agents' forward Q-values using `investigate_Qvalues`. It then calculates 
    the mean Q-value across all positions and extracts the Q-value specifically at the 
    trauma position. The results are combined into a summary table and visualized in a bar plot.

    Args:
        directory (Path): Path to the directory containing the agent trial files.
        agents_to_include (List[str]): List of agent class names to include in the analysis 
            (e.g., ["HealthyAgent", "PTSDAgent"]).
        agent_state (str): Suffix indicating the state of the agents 
            (e.g., "_traumatized", "_exposure_therapy", "_relearned", or "").
        env (Image2VecWrapper): The wrapped environment used to evaluate the agents’ Q-values.
    """

    # determine which places to visit
    places_to_visit = []
    for i in range(1,7):
        for j in range(1,7):
            places_to_visit.append((i,j))
    for i in range(len(env.unwrapped.transient_locations)):
        places_to_visit.remove(tuple(env.unwrapped.transient_locations[i]))
    for i in range(len(env.unwrapped.wall_locations)):
        places_to_visit.remove(tuple(env.unwrapped.wall_locations[i]))
    places_to_visit.remove((6,6))

    state,_ = env.reset()

    # get all the q values in every empty position
    results = []
    for pos in places_to_visit:
        env.unwrapped.agent_start_pos = pos 
        env.render()
        df = investigate_Qvalues(directory, agents_to_include, env, show_plots=False, agent_state=agent_state, single_plot=False)
        df['position'] = [pos] * len(df)
        results.append(df)

    results = pd.concat(results, ignore_index=True)

    # only save the forward q value
    results["qvalues"] = results["qvalues"].apply(lambda x: x[-1])

    # calculate the mean by grouping over the agents and the positions
    grouped_results = results.groupby(["agent", "position"]).mean()

    # Heatmap

    # remove the position of the trauma and save these values indic´vidually 
    mask = grouped_results.index.get_level_values("position") == (3,1)
    trauma_values = grouped_results[mask]
    trauma_values = trauma_values.droplevel("position")
    trauma_values = trauma_values.rename(columns={"qvalues": "trauma position "})

    # remove the trauma values
    cleaned_results = grouped_results[~mask]

    # calculate the mean per agent
    cleaned_results = cleaned_results.groupby(["agent"]).mean()
    cleaned_results = cleaned_results.rename(columns={"qvalues": "overall mean"})

    # join together to a table with a clumun for the mean and another just for the trauma position
    cleaned_results = cleaned_results.join(trauma_values, how="left")
    print(cleaned_results)
    plot_results(directory, cleaned_results, agent_state)


def plot_results(directory: Path, results: pd.DataFrame, agent_state: str):

    folder_path = directory / "img"
    folder_path.mkdir(parents=True, exist_ok=True)

    match agent_state:
        case "_traumatized":
            title = "Comparison of mean Qvalue and Qvalue at trauma position, Split by Agent (after trauma)"
            subpath = "q_values_forward_comparison_after_trauma.png"
        case "_exposure_therapy":
            title = "Comparison of mean Qvalue and Qvalue at trauma position, Split by Agent (after exp. therapy)"
            subpath = "q_values_forward_comparison_after_exposure_therapy.png"
        case "_exposure_therapy_2":
            title = "Comparison of mean Qvalue and Qvalue at trauma position, Split by Agent (after exp. therapy 2)"
            subpath = "q_values_forward_comparison_after_exposure_therapy_2.png"
        case "_relearned":
            title = "Comparison of mean Qvalue and Qvalue at trauma position, Split by Agent (after relearning)"
            subpath = "q_values_forward_comparison_after_relearning.png"
        case "":
            title = "Comparison of mean Qvalue and Qvalue at trauma position, Split by Agent (before trauma)"
            subpath = "q_values_forward_comparison_before_trauma.png"
        case _:
            title = "Comparison of mean Qvalue and Qvalue at trauma position, Split by Agent (unspecified)"
            subpath = "q_values_forward_comparison_unspecified.png"

    # Reshape to long format
    df_long = results.reset_index().melt(id_vars="agent", 
                                    var_name="condition", 
                                    value_name="qvalue")
    # Plot
    plt.figure(figsize=(9,5))
    sns.barplot(data=df_long, x="agent", y="qvalue", hue="condition")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(folder_path /subpath)
    plt.show()

if __name__ == "__main__":
    agents_to_include : List[str] = [
        "HealthyAgent",
        "PTSDAgent",
        "TraumaSynapticDeficitAgent",
    ]

    env = Image2VecWrapper(
                TransientGoals(
                    render_mode="none", transient_reward=0.25, termination_reward=1,
                    transient_locations=[[1,4],[4,2],[5,1]],
                    n_transient_obstacles=0,
                    wall_locations =[[3,2],[3,3],[3,4],[3,5]]
                )
            )
    
    view_Qvalue_differences(DATA_PATH / sys.argv[1], agents_to_include, agent_state= '_exposure_therapy_2', env= env)