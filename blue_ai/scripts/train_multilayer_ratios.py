from copy import deepcopy
import os
from os.path import exists
from pathlib import Path
from typing import List, TypedDict
from pandas.core.frame import itertools
import xxhash
import numpy as np
import polars as pl
from polars import DataFrame
from tqdm import tqdm
import pickle
from blue_ai.agents.agent_classes import (
    BaseAgent,
    HealthyAgent,
    RehabiliationAgent,
    SpineLossDepression,
)
from blue_ai.scripts.constants import DATA_PATH
from blue_ai.scripts.train_agents import run_trial
from blue_ai.scripts.train_ratio_agents import RatioEnvironment


random_ident = lambda: xxhash.xxh32_hexdigest(np.random.rand(100).tobytes())
RUN_ID = ""
pl.enable_string_cache()


def main():
    trials = [
        [
            [RehabiliationAgent()],
            pl.DataFrame({"steps": [10_000, 1000, 10_000], "sum": [8, 8, 1]}),
            30,
        ],
        [
            [HealthyAgent(), SpineLossDepression()],
            pl.DataFrame({"steps": [10_000] * 2, "sum": [1, 1]}),
            40,
        ],
    ]

    tbar = tqdm(
        total=sum(
            stages.select(
                count * len(agents) * (pl.col("sum").cum_prod() * pl.col("steps")).sum()
            ).item()
            for agents, stages, count in trials
        )
    )

    global RUN_ID

    for agents, stages, count in trials:
        RUN_ID = random_ident()
        results, result_agents = [pl.DataFrame()], []

        for a in agents:
            a.metadata["ratios"] = []

        for trial_id in range(count):
            result_agents += ratio_run(
                agents, stages=stages, data=results, trial_id=trial_id, tbar=tbar
            )

        with open(DATA_PATH / f"{RUN_ID}_agents.pickle", "wb") as f:
            pickle.dump(agent, f)


def ratio_run(
    agents: list[BaseAgent],
    stages: pl.DataFrame,
    data: List[DataFrame],
    tbar: tqdm | None = None,
    trial_id: int = 0,
    _current: int = 0,
) -> list[BaseAgent]:
    assert (
        data is None or len(data) == 1
    ), "We only use a list here otherwise pass by ref would not work"
    assert stages.columns == ["steps", "sum"]

    if _current > len(stages) - 1:
        return agents

    steps, sum = stages.row(_current)

    runs = (
        (deepcopy(agent), RatioEnvironment((reward, lava)))
        for reward, lava in gen_ratios(1, sum)
        for agent in agents
    )

    next_agents = []

    for agent, env in runs:
        agent.state_change(stage=_current)

        agent.metadata["ratios"] += env.get_reward_penalty()
        agent.metadata |= dict(
            stage=_current,
            ratio_reward=env.get_reward_penalty()[0],
            ratio_penalty=env.get_reward_penalty()[1],
        )

        d, post_train_agent, _ = run_trial(
            agent=agent,
            env=env,
            steps=steps,
            tbar=tbar,
            trial_id=trial_id,
            polars_ok=True,
            starting_step=agent.metadata.get("end_step") or 0,
            starting_episode_num=agent.metadata.get("end_ep") or 0,
        )

        # Used for start points of these stats in following runs
        agent.metadata["end_step"] = d.select("step").max().item()
        agent.metadata["end_ep"] = d.select("episode").max().item()

        data[0] = pl.concat([data[0], d], how="diagonal_relaxed")

        data[0].write_parquet(
            DATA_PATH / f"{RUN_ID}_full.parquet",
            statistics=False,
            compression="uncompressed",
        )

        next_agents.append(post_train_agent)

    t = ratio_run(
        next_agents,
        stages,
        _current=_current + 1,
        data=data,
        tbar=tbar,
        trial_id=trial_id,
    )

    if _current == 0:
        return t
    return agents + t


def gen_ratios(start: int, end: int):
    return np.column_stack(
        (
            np.arange(start, end + 1),
            np.flip(np.arange(start, end + 1)),
        )
    )


if __name__ == "__main__":
    main()
