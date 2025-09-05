import sys
from pathlib import Path
from typing import List

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS
from blue_ai.scripts.ptsd.train_agents import load_trial, save_trial, run_trial


def induce_traumatic_event(directory: Path, agents_to_include: List[str], trauma_env: Image2VecWrapper, n_trauma_updates: int = 1):
    """
    Induce a traumatic event for the agents by running them in a trauma-inducing environment 
    and saving their updated states.

    This function loads agents from a given directory, runs them in a special environment that 
    simulates trauma, and stores the updated results. For agents of type `TraumaSynapticDeficitAgent`, the optimizer’s 
    weight decay parameter is increased after the trauma induction.

    Args:
        directory (Path): Path to the directory containing the agent trial files.
        agents_to_include (List[str]): List of agent class names to process (e.g., ["HealthyAgent", "PTSDAgent"]).
        trauma_env (Image2VecWrapper): The wrapped environment used to induce trauma in the agents.
        n_trauma_updates (int): Number of trauma updates applied per agent. Defaults to 1.
    """

    # get the file names of the right agents
    files = [
        f"{directory.name}/{agent}_{trial}.pkl"
        for trial in range(N_TRIALS)
        for agent in agents_to_include
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
                g['weight_decay'] = 3e-3

        save_trial(new_results, agent, env, filename)

if __name__ == "__main__":
    agents_to_include : List[str] = [
        "HealthyAgent",
        "PTSDAgent",
        "TraumaSynapticDeficitAgent",
    ]

    # create the environment which is used to induce the ptsd
    traumatic_env = Image2VecWrapper(
                TransientGoals(
                    render_mode="none", transient_reward=0.25, termination_reward=1, agent_start_pos=(3,1),
                    n_transient_obstacles=1, transient_penalty=-100, transient_locations=[[1,4],[4,2],[5,1]], transient_obstacles=[[4,1]],
                    wall_locations =[[3,2],[3,3],[3,4],[3,5],[3,6]], env_name='trauma_env'
                )
            )

    induce_traumatic_event(DATA_PATH / sys.argv[1], agents_to_include, traumatic_env)