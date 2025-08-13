import pandas as pd
from typing import Any, Dict, List, Tuple, TypedDict

from blue_ai.scripts.train_agents import load_trial, run_trial, save_trial
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
import matplotlib.pyplot as plt
from blue_ai.scripts.view_performance import PerformancePlotter
from blue_ai.agents.agent_classes import *
from copy import deepcopy

def main():

    # get the file names of the traumatized agents
    files = [
        filename
        for trial in range(N_TRIALS)
        for filename in [
            f'HealthyAgent_{trial}_traumatized.pkl',
            # f'SpineLossDepression_{trial}.pkl',
            # f'SchizophrenicAgent_{trial}.pkl',
            # f'ReverseImbalanceAgent_{trial}.pkl',
            f'PTSDAgent_{trial}_traumatized.pkl'

        ]
    ]

    # put the agents in the initial environment and have a look at the reward and cumulative reward afterwards?
    initial_env = Image2VecWrapper(
            TransientGoals(
                render_mode="none", transient_reward=0.25, termination_reward=1, n_transient_obstacles=0, n_transient_goals=3,
                #wall_locations =[[3,1],[3,2],[3,3],[3,5],[3,6]]
            )
        )
    
    for i in range(len(files)):
        filename = files[i]
        results, agent, _ = load_trial(DATA_PATH / filename)

        new_results, agent, env = run_trial(agent, initial_env, trial_id=i)

        new_filename = 'harmlessEnv_' + filename
        save_trial(new_results, agent, env, DATA_PATH / new_filename)

    # perform a sort of exposure therapy, where the agents will get put in the environment of the trauma, 
    # but this time without the hazard.
    trauma_env_without_hazard = Image2VecWrapper(
                TransientGoals(
                    render_mode="none", transient_reward=0.25, termination_reward=1, agent_start_pos=(3, 4),
                    transient_locations=[[1,4],[5,3],[5,5]],
                    wall_locations =[[3,1],[3,2],[3,3],[3,5],[3,6]]
                )
            )
    
    for i in range(len(files)):
        filename = files[i]
        results, agent, _ = load_trial(DATA_PATH / filename)

        new_results, agent, env = run_trial(agent, trauma_env_without_hazard, trial_id=i)

        new_filename = 'exposureEnv_' + filename
        save_trial(new_results, agent, env, DATA_PATH / new_filename)

if __name__ == "__main__":
    main()
