import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH
from blue_ai.scripts.ptsd.post_ptsd import post_ptsd
from blue_ai.scripts.ptsd.investigate_Qvalues import investigate_Qvalues

#CURRENTLY NOT IN USE
def view_forward_Qvalues(directory: Path, env: Image2VecWrapper, upper_bound: int = 100):
    """
    Analyze and visualize the evolution of agents' forward Q-values during repeated exposure therapy.

    This function repeatedly applies exposure therapy to agents and computes the average Q-values 
    associated with the 'forward' action for each agent. It tracks how these Q-values change over 
    successive therapy sessions, stores the results in an array, and generates a plot comparing 
    agents’ forward Q-value trends. This helps to evaluate how agents adapt their decision-making 
    during repeated exposure to the therapy environment.

    Args:
        directory (Path): Path to the directory where agent trial data is stored and plots will be saved.
        env (Image2VecWrapper): The environment used for exposure therapy and Q-value analysis.
        upper_bound (int): Upper bound of repeated exposures to simulate. Defaults to 100.
"""

    results = np.zeros((2, upper_bound))

    for i in range(upper_bound):
        if i == 0:
             df = investigate_Qvalues(directory, env, False, "_traumatized", False)
        else: 
            post_ptsd(directory, env, i+1, "_exposure_therapy_forward")
            df = investigate_Qvalues(directory, env, False, "_exposure_therapy_forward", False)

        q_df = pd.DataFrame(df['qvalues'].tolist(), columns=['left', 'right', 'forward'])
        df = pd.concat([df[['agent']], q_df], axis=1)

        # Average per agent
        mean_df = df.groupby('agent').mean().reset_index()
        for j in range(len(mean_df)):
            results[j, i] = mean_df["forward"].values[j]
    
    print(results)
    plot_results(directory, results)
    

def plot_results(directory: Path, results: np.array):

    folder_path = directory / "img"
    folder_path.mkdir(parents=True, exist_ok=True)

    x = np.arange(results.shape[1])

    # Plot each row
    plt.plot(x, results[0], label='HealtyAgent')
    plt.plot(x, results[1], label='PTSDAgent')

    # Add labels, title, and legend
    plt.xlabel('Number of re-exposures')
    plt.ylabel('Forward Q-Value')
    plt.legend()

    # Show plot
    plt.tight_layout
    plt.savefig(folder_path / "q_values_after_n_times_exposure_therapy.png")
    plt.show()

if __name__ == "__main__":
    env = Image2VecWrapper(
                TransientGoals(
                    render_mode="none", transient_reward=0.25, termination_reward=1, agent_start_pos=(3,1),
                    transient_locations=[[1,4],[4,2],[5,1]],
                    n_transient_obstacles=0,
                    wall_locations =[[3,2],[3,3],[3,4],[3,5],[3,6]]
                )
            )
    view_forward_Qvalues(DATA_PATH / sys.argv[1], env, 100)