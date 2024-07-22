from copy import deepcopy
import time
from typing import List, Optional
import numpy as np
import polars as pl
from polars import DataFrame
from tqdm import tqdm
import xxhash

from blue_ai.agents.agent_classes import (
    BaseAgent,
    HealthyAgent,
    RehabiliationAgent,
    SpineLossDepression,
)
from blue_ai.scripts.constants import DATA_PATH
from blue_ai.scripts.train_agents import categorize_pl, run_trial
from blue_ai.scripts.train_ratio_agents import RatioEnvironment


def gen_ratios(start, end):
    forward = np.arange(start, end + 1)
    backward = np.flip(forward)

    return np.column_stack((forward, backward))


N_TRIALS = 5
SUM = 4
STEPS_PER_STAGE = np.array([5, 10]) * 1000


def main():
    simtime = time.time()
    agents = [HealthyAgent(), SpineLossDepression(), RehabiliationAgent()]
    n_stages = len(STEPS_PER_STAGE)

    stages_mult = (SUM + 1) ** (np.arange(n_stages) + 1)
    total = N_TRIALS * len(agents) * np.sum(stages_mult * STEPS_PER_STAGE)
    tbar = tqdm(total=int(total))

    results: List[DataFrame] = []
    result_agents: List[BaseAgent] = []
    for trial_id in range(N_TRIALS):
        result_agents += trial(
            agents,
            data=results,
            limit=n_stages - 1,
            trial_id=trial_id,
            tbar=tbar,
        )
    partials = DATA_PATH / "partial_layered"
    partials.mkdir(exist_ok=True)
    for r in results:
        r = categorize_pl(r)
        r.write_parquet(partials / f"{int(simtime)}.parquet")

    h = xxhash.xxh32()
    combined_results = pl.concat(results, how="diagonal_relaxed")
    h.update(combined_results.hash_rows().to_numpy().tobytes())

    combined_results.write_parquet(
        DATA_PATH / f"ratios_layered_{int(simtime)}_{h.hexdigest()}.parquet"
    )


def trial(
    agents: List[BaseAgent],
    limit=1,
    current=0,
    tbar: Optional[tqdm] = None,
    # We pass this down by reference here, so all levels will append to here
    data: Optional[List[DataFrame]] = None,
    trial_id=0,
):
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

        d, post_train_agent, _ = run_trial(
            agent=agent,
            env=env,
            steps=STEPS_PER_STAGE[current],
            tbar=tbar,
            trial_id=trial_id,
            polars_ok=True,
        )

        d.with_columns(
            stage=pl.lit(current),
            ratio_reward=pl.lit(env.get_reward_penalty()[0]),
            ratio_penalty=pl.lit(env.get_reward_penalty()[1]),
        )
        if data is not None:
            data.append(d)
        next_agents.append(post_train_agent)

    t = trial(next_agents, current=current + 1, data=data, tbar=tbar, trial_id=trial_id)

    if current == 0:
        return t
    return agents + t


if __name__ == "__main__":
    main()
