from functools import lru_cache
from typing import Any, Dict, List, Tuple, TypedDict
import pandas as pd
import pickle
from copy import deepcopy

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from tqdm import tqdm

from blue_ai.scripts.constants import DATA_PATH, N_TRIALS
from blue_ai.agents.agent_classes import *
import polars as pl
import json


def get_len(df, col):
    try:
        return len(df[col].unique())
    except:
        return -1


@lru_cache
def json_cached(x):
    return json.dumps(x)


def categorize(df):
    for c in df.columns:
        if str(df[c].dtype) == "object" and 0 < get_len(df, c) < 100:
            df[c] = pd.Categorical(df[c])


def categorize_pl(df: pl.DataFrame):
    return df.with_columns(
        **{
            col: df[col].cast(pl.Categorical)
            for col in df.select([pl.col(pl.String)]).columns
            if df[col].n_unique() < 50
        }
    )


def run_trial(
    agent: BaseAgent,
    env,
    steps=30000,
    trial_id="",
    tbar=None,
    starting_cumalative_reward=0,
    starting_episode_num=0,
    starting_step=0,
    polars_ok=False,
):
    state, _ = env.reset()
    # setup variables to track progress
    steps_this_episode = 0
    episode_num = starting_episode_num
    cumulative_reward = starting_cumalative_reward

    # setup results dataframe
    results = [None] * steps
    layers = []

    # track agent positions to see if they get stuck
    pos: Dict[Tuple[int, int], int] = {}
    if tbar is not None:
        tbar.set_postfix(
            agent=agent.__class__.__name__, env=env.__class__.__name__, trial=trial_id
        )

    total_reward = sum([o.reward for o in env.unwrapped.obstacles])
    total_penalties = sum([o.reward for o in env.unwrapped.penalties])

    for step in range(steps):
        steps_this_episode += 1

        # record position
        pos[env.unwrapped.agent_pos] = pos.get(env.unwrapped.agent_pos, 0) + 1

        # get & execute action
        action = agent.select_action(state)
        new_state, reward, done, truncated, _ = env.step(action)

        # use this experience to update agent
        agent.update(state, action, reward, new_state, done=False)

        # reset environment if done (ideally env would do this itself)
        if truncated or done:
            state, _ = env.reset()
            episode_num += 1
            steps_this_episode = 0
            total_reward = (
                sum([o.reward for o in env.unwrapped.obstacles])
                + env.unwrapped.termination_reward
            )
            total_penalties = sum([o.reward for o in env.unwrapped.penalties])
        else:
            state = new_state

            total_reward = 0.0
            total_penalties = 0.0

        # add results to the history
        transient_goal = reward == env.unwrapped.transient_reward
        terminal_goal = reward == env.unwrapped.termination_reward
        lava = reward < 0
        stuck = max(pos.values()) > 2000
        cumulative_reward += reward

        flat_layers = pl.Series(
            [
                p.to("cpu", non_blocking=True).detach().numpy().flatten()
                for p in agent.policy_net.parameters()
            ]
        )

        results[step] = (
            {
                "trial_id": trial_id,
                "agent": agent.__class__.__name__,
                "step": step,
                "episode": episode_num,
                "reward": float(reward),
                "cumulative_reward": float(cumulative_reward),
                "terminal_goal": terminal_goal,
                "transient_goal": transient_goal,
                "lava": lava,
                "stuck": stuck,
                # "mean_synapse": next(agent.policy_net.parameters()).mean().item(),
                # "num_pos_synapse": (next(agent.policy_net.parameters()) > 0).sum().item(),
                "total_reward": float(total_reward),
                "total_penalties": float(total_penalties),
                "layer": flat_layers,
            }
            # Add any optional meta data the specific agent may want to provide
            | agent.get_metadata()
        )
        if tbar is not None:
            tbar.update()

    df: pl.DataFrame = pl.DataFrame(
        results,
        schema_overrides={
            "cumulative_reward": pl.Float64,
            "total_reward": pl.Float64,
            "total_penalties": pl.Float64,
        },
    )

    df = categorize_pl(df)

    df = df.with_columns(step=df["step"] + starting_step)

    if not polars_ok:
        df = df.to_pandas()

    return df, agent, env


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
    results = pd.concat(results, ignore_index=True)
    return results


def trial(agent: BaseAgent, env, rep, trial_num, tbar=None, steps=30_000):
    results, agent, env = run_trial(
        agent,
        env,
        steps=steps,
        trial_id=trial_num,
        tbar=tbar,
    )

    filename = (
        DATA_PATH
        / f'{agent.file_display_name()}_{"swapped_" if env.unwrapped.transient_reward > 0.25 else ""}{rep}.pkl'
    )

    save_trial(results, agent, env, filename)
    return trial_num


def main():
    iterations_per_trial = 15_000
    trial_num = 0

    agents: List[BaseAgent] = [
        HealthyAgent(),
        SpineLossDepression(),
        # ContextDependentLearningRate(),
        # HighDiscountRate(),
        # ScaledTargets(),
        # HighExploration(),
        # ShiftedTargets(),
        # PositiveLossAgent(),
        # ReluActivation(),
        # ReluLossActivation(),
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

    # # Setup agent sweep
    # agents += [
    #     PositiveLossAgent(alpha=(2**-x), embed_alpha_in_filename=True)
    #     for x in range(1, 6)
    # ]

    tbar = tqdm(
        total=(len(agents) * len(envs) * N_TRIALS * iterations_per_trial), initial=0
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
                    tbar=tbar,
                    steps=iterations_per_trial,
                )
                trial_num += 1


if __name__ == "__main__":
    main()
