from pathlib import Path
import sys
from typing import List

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS
from blue_ai.scripts.ptsd.train_agents import load_trial, save_trial, run_trial


def test_performance(directory: Path, agents_to_include: List[str], test_env: Image2VecWrapper, agent_state: str, iter_per_trial: int):
    """
    Evaluate the performance of agents in a testing environment without modifying their states.

    This function loads agents at a specified stage (e.g., traumatized, relearned, or after 
    exposure therapy), runs them in a testing environment for a fixed number of iterations, 
    and records their performance results. Unlike training or therapy phases, this step does 
    not update the agents’ internal parameters, serving only as an evaluation of their 
    capabilities at the given state.

    Args:
        directory (Path): Path to the directory containing the agent state files.
        agents_to_include (List[str]): List of agent class names to test 
            (e.g., ["HealthyAgent", "PTSDAgent", "TraumaSynapticDeficitAgent"]).
        test_env (Image2VecWrapper): The environment used to evaluate performance.
        agent_state (str): Identifier for the agent state to load (e.g., "_traumatized", "_relearned").
        iter_per_trial (int): Number of evaluation iterations applied per agent.
    """

    # get the file names of the agents
    files = [
        f"{directory.name}/{agent}_{trial}{agent_state}.pkl"
        for trial in range(N_TRIALS)
        for agent in agents_to_include
    ]

    for i in range(len(files)):
        # load the agent
        filename = files[i]
        _, agent, _ = load_trial(DATA_PATH / filename)

        # test the agents in the testing env
        new_results, agent, env = run_trial(agent, test_env, steps=iter_per_trial, trial_id=i, tbar=None, testing=True)
        print(new_results)

        filename = (DATA_PATH / filename.replace(".pkl", "_testing.pkl",))
        save_trial(new_results, agent, env, filename)
    

if __name__ == "__main__":
    agents_to_include : List[str] = [
        "HealthyAgent",
        "PTSDAgent",
        "TraumaSynapticDeficitAgent",
    ]

    learning_env = Image2VecWrapper(
                TransientGoals(
                    render_mode="human", transient_reward=0.25, termination_reward=1,
                    n_transient_obstacles = 1,
                    wall_locations =[[3,2],[3,3],[3,4],[3,5]],
                )
            )
    
    test_performance(DATA_PATH / sys.argv[1], agents_to_include, learning_env, '_relearned', 5_000)