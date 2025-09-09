from pathlib import Path
import sys
from typing import List, Dict, Tuple

import pandas as pd

from blue_ai.agents.agent_classes import BaseAgent
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS
from blue_ai.scripts.ptsd.train_agents import load_trial, save_trial


def post_ptsd_2(directory: Path, agents_to_include: List[str], exposure_env: Image2VecWrapper, file_ending: str = "_exposure_therapy_2"):
    """
    Apply another approach to exposure therapy to previously traumatized agents by letting
    them update their Qvalues by the actions another Healty Agent chooses.

    This function loads agents that have already undergone trauma and relearning, 
    forces them to perform the updates that the Healthy Agent chooses, 
    and stores the updated results. The process is another approach to therapeutic  
    exposure, allowing agents to adapt to the environment post-trauma.

    Args:
        directory (Path): Path to the directory containing the relearned agent trial files.
        agents_to_include (List[str]): List of agent class names to process 
            (e.g., ["HealthyAgent", "PTSDAgent"]).
        exposure_env (Image2VecWrapper): The wrapped environment used for exposure therapy.
        file_ending (str, optional): Suffix added to the saved trial filenames after exposure therapy. 
            Defaults to "_exposure_therapy_2".
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
        exposure_env.unwrapped.agent_start_pos=(1,1)
        # let the agents follow the lead from a healthy agent
        new_results, agent, env = run_trial_2(directory, agent, exposure_env, steps=1000, trial_id=i, tbar=None)
        print(new_results)

        filename = (DATA_PATH / filename.replace("_relearned.pkl", f"{file_ending}.pkl"))
        save_trial(new_results, agent, env, filename)

def run_trial_2(directory: Path, agent: BaseAgent, env, steps=5000, trial_id="", tbar=None):
    # load the policy agent
    healthy_file = f"{directory.name}/HealthyAgent_0.pkl"
    _, policy_agent, _ = load_trial(DATA_PATH / healthy_file)

    state, _ = env.reset()
    # setup variables to track progress
    episode_num = 0
    cumulative_reward = 0

    # setup results dataframe
    results = [None] * steps

    # track agent positions to see if they get stuck
    pos: Dict[Tuple[int, int], int] = {}
    if tbar is not None:
        tbar.set_postfix(
            agent=agent.__class__.__name__, env=env.__class__.__name__, trial=trial_id
        )

    # actual training loop
    for step in range(steps):

        # record position
        pos[env.unwrapped.agent_pos] = pos.get(env.unwrapped.agent_pos, 0) + 1

        action = policy_agent.select_action(state)
        new_state, reward, done, truncated, _ = env.step(action)
        agent.update(state, action, reward, new_state, done=False)

        # reset environment if done (ideally env would do this itself)
        if truncated or done:
            state, _ = env.reset()
            episode_num += 1
        else:
            state = new_state

        lava = reward < 0
        transient_goal = reward == env.unwrapped.transient_reward
        terminal_goal = reward == env.unwrapped.termination_reward
        stuck = max(pos.values()) > 2000
        cumulative_reward += reward

        results[step] = {
            "trial_id": trial_id,
            "agent": agent.__class__.__name__,
            "step": step,
            "episode": episode_num,
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "terminal_goal": terminal_goal,
            "transient_goal": transient_goal,
            "lava": lava,
            "stuck": stuck,
            "mean_synapse": next(agent.policy_net.parameters()).mean().item(),
            "num_pos_synapse": (next(agent.policy_net.parameters()) > 0).sum().item(),
            'position': tuple(env.unwrapped.agent_pos)
        }

        if tbar is not None:
            tbar.update()

    results = pd.DataFrame(results)
    return results, agent, env
    

if __name__ == "__main__":
    agents_to_include : List[str] = [
        "HealthyAgent",
        "PTSDAgent",
        "TraumaSynapticDeficitAgent",
    ]
    env = Image2VecWrapper(
                TransientGoals(
                    render_mode="none", transient_reward=0.25, termination_reward=1,
                    #agent_start_pos=(3,1),
                    n_transient_obstacles=1, transient_penalty= -1,
                    #transient_locations=[[1,4],[4,2],[5,1]],
                    wall_locations =[[3,2],[3,3],[3,4],[3,5],[3,6]]
                )
            )
    
    post_ptsd_2(DATA_PATH / sys.argv[1], agents_to_include, env)