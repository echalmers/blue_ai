from copy import deepcopy
from typing import NoReturn
import numpy as np
import polars as pl
from polars import DataFrame
from tqdm import tqdm

from blue_ai.agents.agent_classes import (
    BaseAgent,
    HealthyAgent,
    RehabiliationAgent,
    SpineLossDepression,
)
from blue_ai.scripts.constants import DATA_PATH
from blue_ai.scripts.train_agents import run_trial
from blue_ai.scripts.train_ratio_agents import RatioEnvironment


def gen_ratios(start: int, end: int):
    forward = np.arange(start, end + 1)
    backward = np.flip(forward)

    return np.column_stack((forward, backward))


N_TRIALS = 2
SUM = 4
STEPS_PER_STAGE = np.array([5, 5]) * 1000


def main():
    agents: list[BaseAgent] = [
        HealthyAgent(),
        SpineLossDepression(),
        RehabiliationAgent(),
    ]

    n_stages = len(STEPS_PER_STAGE)
    # Find the number of runs per stage
    stages_mult = (SUM + 1) ** (np.arange(n_stages) + 1)
    total = N_TRIALS * len(agents) * np.sum(stages_mult * STEPS_PER_STAGE)

    # We store the previous ratios on the agent itself, so that appears in the data
    # and to make the collection easier
    for a in agents:
        a.metadata["ratios"] = []

    results: list[DataFrame] = []
    result_agents: list[BaseAgent] = []

    tbar = tqdm(total=int(total))

    for trial_id in range(N_TRIALS):
        result_agents += trial(
            agents,
            data=results,
            limit=n_stages - 1,
            trial_id=trial_id,
            tbar=tbar,
        )

    print(len(results) // N_TRIALS)

    combined_results = pl.concat(results, how="diagonal_relaxed")
    combined_results.write_parquet(DATA_PATH / "ratios_layered.parquet")


def trial(
    agents: list[BaseAgent],
    limit: int = 1,
    current: int = 0,
    tbar: tqdm | None = None,
    # We pass this down by reference here, so all levels will append to here
    data: list[DataFrame] | None = None,
    trial_id: int = 0,
) -> list[BaseAgent]:
    assert len(STEPS_PER_STAGE) - 1 >= limit

    if current > limit:
        return agents

    runs = (
        (deepcopy(agent), RatioEnvironment((reward, lava)))
        for reward, lava in gen_ratios(0, SUM)
        for agent in agents
    )

    next_agents = []

    for agent, env in runs:
        agent.state_change(stage=current)
        agent.metadata["ratios"] += env.get_reward_penalty()
        agent.metadata |= dict(
            stage=current,
            ratio_reward=env.get_reward_penalty()[0],
            ratio_penalty=env.get_reward_penalty()[1],
        )

        d, post_train_agent, _ = run_trial(
            agent=agent,
            env=env,
            steps=STEPS_PER_STAGE[current],
            tbar=tbar,
            trial_id=trial_id,
            polars_ok=True,
        )

        if data is not None:
            data.append(d)
        next_agents.append(post_train_agent)

    t = trial(
        next_agents,
        current=current + 1,
        data=data,
        tbar=tbar,
        trial_id=trial_id,
    )

    if current == 0:
        return t
    return agents + t


if __name__ == "__main__":
    main()
