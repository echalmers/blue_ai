import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import sys
from pathlib import Path

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.train_agents import load_trial
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS



def investigate_Qvalues(directory: Path, env: Image2VecWrapper, show_plots: bool, trauma: bool):
    # get the file names of the agents
    files = [
        filename
        for trial in range(N_TRIALS)
        for filename in [
            f'{directory.name}/HealthyAgent_{trial}{"_traumatized" if trauma else ""}.pkl',
            f'{directory.name}/PTSDAgent_{trial}{"_traumatized" if trauma else ""}.pkl'

        ]
    ]

    # make sure that the hazard is removed
    env.unwrapped.n_transient_obstacles = 0
    env.unwrapped.transient_obstacles = None

    results = []
    for i in range(len(files)):
        # load the agent
        filename = files[i]
        _, agent, _ = load_trial(DATA_PATH / filename)
        state, _ = env.reset()
        env.render() # it is currently not working when being called from the execute script?
        result = {
            "agent": agent.__class__.__name__,
            "qvalues": np.array(agent.get_action_values(state), dtype=np.float32)
        }
        results.append(result)
        #print(f"Agent: {agent.__class__.__name__}, Qvalues: {qvalues}")
    
    results = pd.DataFrame(results)
    print(results)
    plotting(results, directory, show_plots, trauma)
    

def plotting(df: pd.DataFrame, directory: Path, show_plots: bool, trauma: bool):

    folder_path = directory / "img"
    folder_path.mkdir(parents=True, exist_ok=True)

    # Expand Q_values into columns
    q_df = pd.DataFrame(df['qvalues'].tolist(), columns=['left', 'right', 'forward', 'stand'])
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
    plt.suptitle(f"Average Q-values per Action {'after' if trauma else 'before'} trauma, Split by Agent")
    plt.savefig(folder_path / f"q_values_{"after" if trauma else "before"}_trauma.png")
    if show_plots:
        plt.show()
    

if __name__ == "__main__":
    # place them in the exact position where they they would have gotten traumatized but without hazard
    env = Image2VecWrapper(
                TransientGoals(
                    render_mode="human", transient_reward=0.25, termination_reward=1, agent_start_pos=(3,1),
                    transient_locations=[[1,4],[4,2],[5,1]],
                    n_transient_obstacles=0,
                    wall_locations =[[3,2],[3,3],[3,4],[3,5],[3,6]], env_name='trauma_env'
                )
            )
    investigate_Qvalues(DATA_PATH / sys.argv[1], env, trauma=True)
