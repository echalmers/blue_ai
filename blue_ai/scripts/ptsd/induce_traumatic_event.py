import sys
from pathlib import Path

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS
from blue_ai.scripts.ptsd.train_agents import load_trial, save_trial, run_trial


def induce_traumatic_event(directory: Path, trauma_env: Image2VecWrapper, n_trauma_updates: int = 1):

    # get the file names of the right agents
    files = [
        filename
        for trial in range(N_TRIALS)
        for filename in [
            f'{directory.name}/HealthyAgent_{trial}.pkl',
            f'{directory.name}/PTSDAgent_{trial}.pkl',
            f'{directory.name}/TraumaSynapticDeficitAgent_{trial}.pkl',
        ]
    ]

    # induce the trauma to each agent trial independently
    for i in range(len(files)):
        filename = files[i]
        _, agent, _ = load_trial(DATA_PATH / filename)

        new_results, agent, env = run_trial(agent, trauma_env, steps=n_trauma_updates, trial_id=i, tbar=None, trauma=True)
        print(new_results)
        
        filename = (DATA_PATH / filename.replace(".pkl", "_traumatized.pkl"))

        # change the weight decay for the TraumaSynapticDeficitAgent after it got induced with the trauma
        if agent.__class__.__name__ == "TraumaSynapticDeficitAgent":
            for g in agent.optimizer.param_groups:
                g['weight_decay'] = 1e-3

        save_trial(new_results, agent, env, filename)

if __name__ == "__main__":
    # create the environment which is used to induce the ptsd
    traumatic_env = Image2VecWrapper(
                TransientGoals(
                    render_mode="none", transient_reward=0.25, termination_reward=1, agent_start_pos=(3,1),
                    n_transient_obstacles=1, transient_penalty=-100, transient_locations=[[1,4],[4,2],[5,1]], transient_obstacles=[[4,1]],
                    wall_locations =[[3,2],[3,3],[3,4],[3,5],[3,6]], env_name='trauma_env'
                )
            )

    induce_traumatic_event(DATA_PATH / sys.argv[1], traumatic_env)

