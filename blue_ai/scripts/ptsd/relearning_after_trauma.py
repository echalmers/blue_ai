from pathlib import Path
import sys

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS
from blue_ai.scripts.ptsd.train_agents import load_trial, save_trial, run_trial


def relearning_after_trauma(directory: Path, relearning_env: Image2VecWrapper, iter_per_trial: int):

    # get the file names of the agents
    files = [
        filename
        for trial in range(N_TRIALS)
        for filename in [
            f'{directory.name}/HealthyAgent_{trial}_traumatized.pkl',
            f'{directory.name}/PTSDAgent_{trial}_traumatized.pkl',
            f'{directory.name}/TraumaSynapticDeficitAgent_{trial}_traumatized.pkl'
        ]
    ]

    # training loop
    for i in range(len(files)):
        # load the agent
        filename = files[i]
        _, agent, _ = load_trial(DATA_PATH / filename)
        # test the agents in the relearning env
        new_results, agent, env = run_trial(agent, relearning_env, steps=iter_per_trial, trial_id=i, tbar=None)
        print(new_results)
        # save the results
        filename = (DATA_PATH / filename.replace("_traumatized.pkl", "_relearned.pkl",))
        save_trial(new_results, agent, env, filename)
    

if __name__ == "__main__":
    relearning_env = Image2VecWrapper(
                TransientGoals(
                    render_mode="none", transient_reward=0.25, termination_reward=1,
                    n_transient_obstacles = 1,
                    wall_locations =[[3,2],[3,3],[3,4],[3,5],[3,6]],
                )
            )
    
    relearning_after_trauma(DATA_PATH / sys.argv[1], relearning_env, 80_000)

