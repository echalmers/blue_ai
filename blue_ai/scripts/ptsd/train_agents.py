from typing import Dict, List, Tuple
import sys
import pickle
from copy import deepcopy
from tqdm import tqdm

from torch import nn
import pandas as pd

from blue_ai.agents.agent_classes import BaseAgent, HealthyAgent, PTSDAgent
from blue_ai.envs.transient_goals import TransientGoals, Actions
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS


# introduce a fourth action 'hide'
ptsd_network = nn.Sequential(
    nn.Flatten(1, -1),
    nn.Linear(100, 25), nn.Tanh(),
    nn.Linear(25, 4)
)


def run_trial(agent: BaseAgent, env, steps=30000, trial_id="", tbar=None, trauma = False, exposure_therapy = False):
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

        # get & execute action
        if trauma or exposure_therapy:
            action = Actions.forward
            new_state, reward, done, truncated, _ = env.step(action)
            agent.update_single(state, action, reward, new_state, done=False)
            done = True
        else:
            action = agent.select_action(state)
            new_state, reward, done, truncated, _ = env.step(action)
            agent.update(state, action, reward, new_state, done=False)

        # reset environment if done (ideally env would do this itself)
        if truncated or done:
            state, _ = env.reset()
            episode_num += 1
        else:
            state = new_state

        # if there is a hiding penalty, account for it in the hazard detection
        if env.unwrapped.hiding_penalty:
            lava = reward < -env.unwrapped.hiding_penalty_value
        else: 
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


def save_trial(results, agent, env, filename):
    with open(filename, "wb") as f:
        pickle.dump({"results": results, "agent": agent, "env": env}, f)


def load_trial(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["results"], data["agent"], data["env"]


def load_dataset(filename_patterns, return_agents=False):
    if isinstance(filename_patterns, str):
        filename_patterns = [filename_patterns]
    results = []
    agents = dict()
    for pattern in filename_patterns:
        files = list(DATA_PATH.glob(pattern))
        for filename in tqdm(files, leave=False, total=len(files)):
            this_result, agent, _ = load_trial(filename)
            this_result["agent"] = (
                agent.display_name
                if hasattr(agent, "display_name")
                else agent.__class__.__name__
            )
            this_result["filename"] = filename
            results.append(this_result)

            if return_agents:
                agents[filename] = agent
    results = pd.concat(results, ignore_index=True)

    if return_agents:
        return results, agents
    return results


def trial(agent: BaseAgent, env, rep, trial_num, directory, tbar=None, steps=30_000):
    results, agent, env = run_trial(
        agent,
        env,
        steps=steps,
        trial_id=trial_num,
        tbar=tbar,
    )

    filename = (f'{directory}/{agent.file_display_name()}_{rep}.pkl')
    

    save_trial(results, agent, env, filename)
    return trial_num


def train_agents(agents: List[BaseAgent], envs: List[Image2VecWrapper], iter_per_trial: int, directory: str):
    trial_num = 0
    tbar = tqdm(
        total=(len(agents) * N_TRIALS * iter_per_trial), initial=0
    )

    for rep in range(N_TRIALS):
        for env in envs:
            for agent in agents:
                tbar.set_postfix(
                    agent=agent.__class__.__name__, env=env.__class__.__name__, rep=rep
                )
                trial(
                    deepcopy(agent),
                    env,
                    rep,
                    trial_num,
                    directory,
                    tbar=tbar,
                    steps=iter_per_trial
                )
                trial_num += 1

if __name__ == "__main__":
    iterations_per_trial = 80_000
    directory = DATA_PATH / sys.argv[1]
    directory.mkdir(parents=True, exist_ok=True)

    agents: List[BaseAgent] = [
        HealthyAgent(network= ptsd_network),
        PTSDAgent(network= ptsd_network)
    ]

    envs = [
        Image2VecWrapper(
                TransientGoals(
                    render_mode="none", transient_reward=0.25, termination_reward=1,
                    n_transient_obstacles = 1,
                    wall_locations =[[3,2],[3,3],[3,4],[3,5],[3,6]],
                    #hiding_penalty=True, hiding_penalty_value = 0.25
                    # see_through_walls = False IS NOT WORKING?
                )
            )
    ]
    
    train_agents(agents, envs, iterations_per_trial, directory)
