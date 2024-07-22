from typing import List, Tuple

import polars as pl
from blue_ai.agents.agent_classes import (
    BaseAgent,
    HealthyAgent,
    RehabiliationAgent,
    SpineLossDepression,
)
from blue_ai.envs.custom_wrappers import Image2VecWrapper
import numpy as np

from blue_ai.envs.transient_goals import TransientGoals
from copy import deepcopy
from blue_ai.scripts.constants import DATA_PATH
from blue_ai.scripts.train_agents import categorize_pl, run_trial

from tqdm import tqdm


class RatioEnvironment(Image2VecWrapper):
    def get_reward_penalty(self):
        return self.ratio

    def __repr__(self):
        return f"{self.name} {hex(id(self))}"

    def __init__(self, ratio: Tuple[int, int]):
        super().__init__(
            TransientGoals(
                render_mode="none",
                n_transient_goals=ratio[0],
                n_transient_obstacles=ratio[1],
            ),
        )
        self.ratio = ratio
        self.name = f"Ratio(Rewards: {self.ratio[0]}, Lava: {self.ratio[1]})"


def ratio_environments(start, end):
    forward = np.arange(start, end)
    backward = np.flip(forward)

    ratio_pairs = np.column_stack((forward, backward))

    for pair in ratio_pairs:
        yield RatioEnvironment(pair)


def main():
    N_TRIALS = 5
    RATIO_SUM = 8
    # Start in a 1:1 environment branch into each of the ratios in (total - k, k) for k {0, total}

    STEPS_PER_STAGE = np.array([20_000, 20_000, 20_000])

    balanced = RatioEnvironment((RATIO_SUM // 2, RATIO_SUM // 2))

    agents: List[BaseAgent] = [
        HealthyAgent(),
        SpineLossDepression(),
        RehabiliationAgent(),
    ]

    total_branched_stepts = RATIO_SUM * sum(STEPS_PER_STAGE[1:])
    tbar = tqdm(
        total=N_TRIALS * len(agents) * (STEPS_PER_STAGE[0] + total_branched_stepts)
    )

    trials = (
        (deepcopy(agent), trial_id) for agent in agents for trial_id in range(N_TRIALS)
    )

    results: List[pl.DataFrame] = []

    for agent, trial_id in trials:

        agent.state_change(stage=0)
        r0, agent, _ = run_trial(
            env=deepcopy(balanced),
            agent=agent,
            steps=STEPS_PER_STAGE[0],
            tbar=tbar,
            trial_id=trial_id,
            polars_ok=True,
        )

        results.append(
            r0.with_columns(
                state=pl.lit(1),
                ratio_reward=balanced.get_reward_penalty()[0],
                ratio_penalty=balanced.get_reward_penalty()[1],
                name=pl.lit("Pre-Treatment"),
            )
        )

        # Start rehab
        agent.state_change(stage=1)

        for e in ratio_environments(1, RATIO_SUM + 1):
            starting_episode_num = r0["episode"].max()
            starting_step = r0["step"].max()

            r, r_agent, _ = run_trial(
                env=e,
                agent=deepcopy(agent),
                steps=STEPS_PER_STAGE[1],
                tbar=tbar,
                trial_id=trial_id,
                polars_ok=True,
                starting_step=starting_step,
                starting_episode_num=starting_episode_num,
            )

            results.append(
                r.with_columns(
                    state=pl.lit(1),
                    ratio_reward=e.get_reward_penalty()[0],
                    ratio_penalty=e.get_reward_penalty()[1],
                    name=pl.lit(e.name),
                )
            )

            starting_episode_num = r["episode"].max()
            starting_step = r["step"].max()

            recovery_env = deepcopy(balanced)
            recovery_env.name = e.name

            r, _, _ = run_trial(
                env=deepcopy(balanced),
                agent=r_agent,
                steps=STEPS_PER_STAGE[2],
                tbar=tbar,
                trial_id=trial_id,
                polars_ok=True,
                starting_step=starting_step,
                starting_episode_num=starting_episode_num,
            )

            results.append(
                r.with_columns(
                    state=pl.lit(2),
                    ratio_reward=recovery_env.get_reward_penalty()[0],
                    ratio_penalty=recovery_env.get_reward_penalty()[1],
                    name=pl.lit(e.name),
                    is_recovery=pl.lit(True),
                )
            )

    joined_results = pl.concat(results, how="diagonal")
    joined_results = categorize_pl(joined_results)

    joined_results.write_parquet(DATA_PATH / "ratios.parquet")

    tbar.close()


if __name__ == "__main__":
    main()
