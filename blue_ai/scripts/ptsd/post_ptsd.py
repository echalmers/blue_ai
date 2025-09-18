from pathlib import Path
import sys
from typing import List

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS
from blue_ai.scripts.ptsd.train_agents import load_trial, save_trial, run_trial

# CURRENTLY NOT IN USE 
def post_ptsd(directory: Path, agents_to_include: List[str], exposure_env: Image2VecWrapper, n_exposure_updates: int, file_ending: str = "_exposure_therapy"):
    """
    Apply exposure therapy to previously traumatized agents by running them in a 
    controlled exposure environment and saving their updated states.

    This function loads agents that have already undergone trauma and relearning, 
    forces them through an exposure environment for a specified number of updates, 
    and stores the updated results. The process simulates therapeutic exposure, 
    allowing agents to adapt to the environment post-trauma.

    Args:
        directory (Path): Path to the directory containing the relearned agent trial files.
        agents_to_include (List[str]): List of agent class names to process 
            (e.g., ["HealthyAgent", "PTSDAgent"]).
        exposure_env (Image2VecWrapper): The wrapped environment used for exposure therapy.
        n_exposure_updates (int): Number of exposure updates applied per agent.
        file_ending (str, optional): Suffix added to the saved trial filenames after exposure therapy. 
            Defaults to "_exposure_therapy".
    """

    # get the file names of the agents
    files = [
        f"{directory.name}/{agent}_{trial}_relearned.pkl"
        for trial in range(N_TRIALS)
        for agent in agents_to_include
    ]

    # load agents and place them in the desired exposure therapy environment
    for i in range(len(files)):
        filename = files[i]
        _, agent, _ = load_trial(DATA_PATH / filename)
        exposure_env.unwrapped.agent_start_pos=(3,1)
        # force the agent through the exposure experience
        new_results, agent, env = run_trial(agent, exposure_env, steps=n_exposure_updates, trial_id=i, tbar=None, exposure_therapy = True)
        print(new_results)

        filename = (DATA_PATH / filename.replace("_relearned.pkl", f"{file_ending}.pkl"))
        save_trial(new_results, agent, env, filename)
    

if __name__ == "__main__":
    agents_to_include : List[str] = [
        "HealthyAgent",
        "PTSDAgent",
        "TraumaSynapticDeficitAgent",
    ]
    env = Image2VecWrapper(
                TransientGoals(
                    render_mode="none", transient_reward=0.25, termination_reward=1,
                    agent_start_pos=(3,1),
                    n_transient_obstacles=0,
                    transient_locations=[[1,4],[4,2],[5,1]],
                    wall_locations =[[3,2],[3,3],[3,4],[3,5],[3,6]]
                )
            )
    
    post_ptsd(DATA_PATH / sys.argv[1], agents_to_include, env, 3)