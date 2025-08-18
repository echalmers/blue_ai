import pandas as pd
from typing import Dict, Tuple
import sys
from pathlib import Path

from blue_ai.scripts.train_agents import load_trial, save_trial
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.agents.agent_classes import BaseAgent


def induce_traumatic_event(directory: Path, trauma_env: Image2VecWrapper):

    # get the file names of the right agents
    files = [
        filename
        for trial in range(N_TRIALS)
        for filename in [
            f'{directory.name}/HealthyAgent_{trial}.pkl',
            f'{directory.name}/PTSDAgent_{trial}.pkl'

        ]
    ]

    # induce the trauma to each agent trial independently
    for i in range(len(files)):
        filename = files[i]
        results, agent, env = load_trial(DATA_PATH / filename)

        new_results, agent, env = run_trauma(agent, trauma_env, n_trauma_updates=1, trial_id=i)
        print(new_results)
        
        filename = ( DATA_PATH / filename.replace(".pkl", "_traumatized.pkl") )
        
        # save the new agent with the old results
        save_trial(results, agent, env, filename)


def run_trauma(agent: BaseAgent, env: Image2VecWrapper, n_trauma_updates: int, trial_id=""):
    state, _ = env.reset()
    episode_num = 0
    cumulative_reward = 0
    reward = 0
    step = -1

    results = []

    # track agent positions to see if they get stuck
    pos: Dict[Tuple[int, int], int] = {}

    # run loop until the agent gets traumatized
    while True:
        step +=1
        # record position
        pos[env.unwrapped.agent_pos] = pos.get(env.unwrapped.agent_pos, 0) + 1

        action = agent.select_action(state)
        new_state, reward, done, truncated, _ = env.step(action)

        # only update the traumatic event
        if reward == env.unwrapped.transient_penalty:
            for _ in range(n_trauma_updates):
                agent.update_single(state, action, reward, new_state, done=False)
        
        if truncated or done:
            state, _ = env.reset()
            episode_num +=1
        else:
            state = new_state

        # add results to the history
        transient_goal = reward == env.unwrapped.transient_reward
        terminal_goal = reward == env.unwrapped.termination_reward
        lava = reward < 0
        stuck = max(pos.values()) > 2000
        cumulative_reward += reward

        result =  {
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

        results.append(result)

        # break once the agent is traumatized
        if reward == env.unwrapped.transient_penalty:
            break
    
    results = pd.DataFrame(results)
    return results, agent, env

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

