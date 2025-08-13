import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple, TypedDict

from blue_ai.scripts.train_agents import load_trial, run_trial, save_trial
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS, FIGURE_PATH
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.view_performance import PerformancePlotter
from blue_ai.agents.agent_classes import *
from copy import deepcopy

def main():
    # MAYBE MACH DAS DYNAMIC WO ICH NUR FILES ÜBERGEBE, DANN BRAUCH ICH NICHTS DOPPELT

    # get the file names of the untraumatized agents
    
    files = [
        filename
        for trial in range(N_TRIALS)
        for filename in [
            f'top_opening_tanh/HealthyAgent_{trial}_tanh_traumatized.pkl',
            # f'SpineLossDepression_{trial}.pkl',
            # f'SchizophrenicAgent_{trial}.pkl',
            # f'ReverseImbalanceAgent_{trial}.pkl',
            f'top_opening_tanh/PTSDAgent_{trial}_tanh_traumatized.pkl'

        ]
    ]

    # place them in the exact position where they they would have gotten traumatized but without hazard
    env = Image2VecWrapper(
                TransientGoals(
                    render_mode="none", transient_reward=0.25, termination_reward=1, agent_start_pos=(3,1),
                    transient_locations=[[1,4],[4,2],[5,1]],
                    wall_locations =[[3,2],[3,3],[3,4],[3,5],[3,6]], env_name='trauma_env'
                )
            )
    results = []
    for i in range(len(files)):
        # load the agent
        filename = files[i]
        _, agent, _ = load_trial(DATA_PATH / filename)
        state, _ = env.reset()
        result = {
            "agent": agent.__class__.__name__,
            "qvalues": np.array(agent.get_action_values(state), dtype=np.float32)
        }
        results.append(result)
        #print(f"Agent: {agent.__class__.__name__}, Qvalues: {qvalues}")
    
    results = pd.DataFrame(results)
    print(results)
    plotting(results)
    # dont let them move, just check q values

    # save them 

    # get the file names of the traumatized agents

    # place them in the exact position of the trauma

    # dont let them move, just check q values

    # save them 

    # compare and plot the q values

def plotting(df: pd.DataFrame):

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
    plt.suptitle("Average Q-values per Action, Split by Agent")
    plt.savefig(FIGURE_PATH / "q_values_after_trauma_tanh.png")
    plt.show()

if __name__ == "__main__":
    main()
