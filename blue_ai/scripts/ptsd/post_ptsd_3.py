from pathlib import Path
import sys
from typing import List

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS
from blue_ai.scripts.ptsd.train_agents import load_trial, save_trial, run_trial


def post_ptsd_3(directory: Path, agents_to_include: List[str], env: Image2VecWrapper, iter_per_trial: int, file_ending: str = "_connectivity_restoration"):
    """
    Apply another approach to exposure therapy to previously traumatized agents by reversing the 
    connectivity impairment (lowering the weight decay).

    This function loads agents that have already undergone trauma and relearning, 
    lowers the weight decay of the PTSDAgents and the TraumaSynapticDeficitAgent, 
    and stores the updated results. The process is another approach to compare if restoring 
    the neural connectivity would lead to similar results to agents who never had a 
    connectivity impairment in the first place.

    Args:
        directory (Path): Path to the directory containing the relearned agent trial files.
        agents_to_include (List[str]): List of agent class names to process 
            (e.g., ["HealthyAgent", "PTSDAgent"]).
        env (Image2VecWrapper): The wrapped environment used for therapy.
        file_ending (str, optional): Suffix added to the saved trial filenames after therapy. 
            Defaults to "_connectivity_restoration".
    """

    # get the file names of the agents
    files = [
        f"{directory.name}/{agent}_{trial}_relearned.pkl"
        for trial in range(N_TRIALS)
        for agent in agents_to_include
    ]

    # load agents and place them in the desired environment
    for i in range(len(files)):
        filename = files[i]
        _, agent, _ = load_trial(DATA_PATH / filename)

        # make sure the starting position from the environment is (1,1)
        env.unwrapped.agent_start_pos=(1,1)

        # restore the connectivity
        if agent.__class__.__name__ != "HealthyAgent":
            for g in agent.optimizer.param_groups:
                g['weight_decay'] = 1e-5
                
        
        new_results, agent, env = run_trial(agent, env, steps=iter_per_trial, trial_id=i, tbar=None)
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
                n_transient_obstacles = 1,
                wall_locations =[[3,2],[3,3],[3,4],[3,5]]#,[3,6]],
            )
        )
    
    post_ptsd_3(DATA_PATH / sys.argv[1], agents_to_include, env, 40_000)