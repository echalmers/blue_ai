from pathlib import Path
import sys
from typing import List

from blue_ai.agents.agent_classes import BaseAgent
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS
from blue_ai.scripts.ptsd.train_agents import load_trial, save_trial, run_trial


def relearning_after_trauma(directory: Path, agents_to_include: List[BaseAgent], relearning_env: Image2VecWrapper, iter_per_trial: int):
    """
    Allow traumatized agents to relearn in a safe environment and save their updated states.

    This function loads agents that have previously undergone trauma, runs them in a 
    controlled relearning environment for a specified number of iterations, and saves the 
    updated results. The process simulates recovery and adaptation after trauma exposure, 
    enabling agents to possibly regain performance to a certain degree in a non-traumatic setting.

    Args:
        directory (Path): Path to the directory containing the traumatized agent trial files.
        agents_to_include (List[BaseAgent]): List of agent class names to process 
            (e.g., ["HealthyAgent", "PTSDAgent", "TraumaSynapticDeficitAgent"]).
        relearning_env (Image2VecWrapper): The wrapped environment used for relearning after trauma.
        iter_per_trial (int): Number of training iterations applied per agent during relearning.
    """

    # get the file names of the agents
    files = [
        f"{directory.name}/{agent}_{trial}_traumatized.pkl"
        for trial in range(N_TRIALS)
        for agent in agents_to_include
    ]

     # load agents and place them in the desired relearning environment
    for i in range(len(files)):
        filename = files[i]
        _, agent, _ = load_trial(DATA_PATH / filename)

        new_results, agent, env = run_trial(agent, relearning_env, steps=iter_per_trial, trial_id=i, tbar=None)
        print(new_results)

        filename = (DATA_PATH / filename.replace("_traumatized.pkl", "_relearned.pkl",))
        save_trial(new_results, agent, env, filename)
    

if __name__ == "__main__":
    agents_to_include : List[str] = [
        "HealthyAgent",
        "PTSDAgent",
        "TraumaSynapticDeficitAgent",
    ]
    relearning_env = Image2VecWrapper(
                TransientGoals(
                    render_mode="none", transient_reward=0.25, termination_reward=1,
                    n_transient_obstacles = 1,
                    wall_locations =[[3,2],[3,3],[3,4],[3,5],[3,6]],
                )
            )
    
    relearning_after_trauma(DATA_PATH / sys.argv[1], agents_to_include, relearning_env, 80_000)