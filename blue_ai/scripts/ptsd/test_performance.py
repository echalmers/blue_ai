from pathlib import Path
import sys
from typing import List

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS
from blue_ai.scripts.ptsd.train_agents import load_trial, save_trial, run_trial


def test_performance(directory: Path, agents_to_include: List[str], test_env: Image2VecWrapper, agent_state: str, iter_per_trial: int):

    # get the file names of the agents
    files = [
        f"{directory.name}/{agent}_{trial}{agent_state}.pkl"
        for trial in range(N_TRIALS)
        for agent in agents_to_include
    ]

    # training loop
    for i in range(len(files)):
        # load the agent
        filename = files[i]
        _, agent, _ = load_trial(DATA_PATH / filename)
        # test the agents in the testing env
        new_results, agent, env = run_trial(agent, test_env, steps=iter_per_trial, trial_id=i, tbar=None, testing=True)
        print(new_results)
        # save the results
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
                    render_mode="none", transient_reward=0.25, termination_reward=1,
                    n_transient_obstacles = 1,
                    wall_locations =[[3,2],[3,3],[3,4],[3,5],[3,6]],
                )
            )
    
    test_performance(DATA_PATH / sys.argv[1], agents_to_include, learning_env, '_exposure_therapy_2', 5_000)

