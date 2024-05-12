import pandas as pd
import pickle
import argparse

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from tqdm import tqdm

from constants import DATA_PATH, N_TRIALS

if __name__ == "__main__":
    from blue_ai.agents.agent_classes import (
        HealthyAgent,
        SpineLossDepression,
        ContextDependentLearningRate,
        HighDiscountRate,
        ScaledTargets,
        ShiftedTargets,
        HighExploration,
    )


def run_trial(agent, env, steps=30000, trial_id="", tbar=None):
    state, _ = env.reset()

    # setup variables to track progress
    steps_this_episode = 0
    episode_num = 0
    cumulative_reward = 0

    # setup results dataframe
    results = [None] * steps

    # track agent positions to see if they get stuck
    pos = {}
    steps_iter = tqdm(range(steps), leave=False)
    steps_iter.set_postfix(
        agent=agent.__class__.__name__, env=env.__class__.__name__, trial=trial_id
    )

    for step in steps_iter:
        steps_this_episode += 1

        # record position
        pos[env.unwrapped.agent_pos] = pos.get(env.unwrapped.agent_pos, 0) + 1

        # get & execute action
        action = agent.select_action(state)
        new_state, reward, done, _, _ = env.step(action)

        # use this experience to update agent
        agent.update(state, action, reward, new_state, done=False)

        # reset environment if done (ideally env would do this itself)
        if done or steps_this_episode > 500:
            state, _ = env.reset()
            episode_num += 1
            steps_this_episode = 0
        else:
            state = new_state

        # add results to the history
        transient_goal = reward == env.unwrapped.transient_reward
        terminal_goal = reward == env.unwrapped.termination_reward
        lava = reward < 0
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
        }

        if tbar is not None:
            tbar.update()

    results = pd.DataFrame(results)
    steps_iter.close()
    return results, agent, env


def save_trial(results, agent, env, filename):
    with open(filename, "wb") as f:
        pickle.dump({"results": results, "agent": agent, "env": env}, f)


def load_trial(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["results"], data["agent"], data["env"]


def load_dataset(filename_patterns):
    if isinstance(filename_patterns, str):
        filename_patterns = [filename_patterns]
    results = []
    for pattern in filename_patterns:
        print(len(list(DATA_PATH.glob(pattern))))
        for filename in DATA_PATH.glob(pattern):
            this_result, agent, _ = load_trial(filename)
            this_result["agent"] = agent.display_name
            results.append(this_result)
    results = pd.concat(results, ignore_index=True)
    return results


def trial(agent, env, rep, trial_num, tbar=None):
    results, agent, env = run_trial(
        agent, env, steps=30_000, trial_id=trial_num, tbar=tbar
    )

    filename = (
        DATA_PATH
        / f'{agent.__class__.__name__}_{"swapped_" if env.unwrapped.transient_reward > 0.25 else ""}{rep}.pkl'
    )
    save_trial(results, agent, env, filename)
    return trial_num


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("retrain_all")

    trial_num = 0

    agents = [
        HealthyAgent(),
        SpineLossDepression(),
        # ContextDependentLearningRate(),
        # HighDiscountRate(),
        # ScaledTargets(),
        # HighExploration(),
        # ShiftedTargets(),
    ]
    envs = [
        Image2VecWrapper(
            TransientGoals(
                render_mode="none", transient_reward=0.25, termination_reward=1
            )
        ),
        # swapped reward structure
        # Image2VecWrapper(TransientGoals(render_mode='none', transient_reward=1, termination_reward=0.25)),
    ]

    pbar = tqdm(total=(len(agents) * len(envs) * N_TRIALS), initial=0)
    tbar = tqdm(total=(len(agents) * len(envs) * N_TRIALS) * 30_000, initial=0)

    for rep in range(N_TRIALS):
        for env in envs:
            for agent in agents:
                trial(agent, env, rep, trial_num, tbar=tbar)
                pbar.update()
                trial_num += 1

    pbar.close()


if __name__ == "__main__":
    main()
