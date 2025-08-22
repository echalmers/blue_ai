from pathlib import Path
import sys

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS
from blue_ai.scripts.ptsd.train_agents import load_trial, save_trial, run_trial


def test_performance(directory: Path, test_env: Image2VecWrapper, agent_state: str, iter_per_trial: int):

    # get the file names of the agents
    files = [
        filename
        for trial in range(N_TRIALS)
        for filename in [
            f'{directory.name}/HealthyAgent_{trial}{agent_state}.pkl',
            f'{directory.name}/PTSDAgent_{trial}{agent_state}.pkl'
        ]
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
    learning_env = Image2VecWrapper(
                TransientGoals(
                    render_mode="none", transient_reward=0.25, termination_reward=1,
                    n_transient_obstacles = 1,
                    wall_locations =[[3,2],[3,3],[3,4],[3,5],[3,6]],
                    # hiding_penalty=True, hiding_penalty_value = 0.25
                    # see_through_walls = False IS NOT WORKING?
                )
            )
    
    test_performance(DATA_PATH / sys.argv[1], learning_env, '', 80_000)

