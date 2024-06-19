from numpy import tile
from torch import polar
from tqdm import tqdm
from blue_ai.agents.agent_classes import (
    BaseAgent,
    HealthyAgent,
    RehabiliationAgent,
    SpineLossDepression,
)
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS
from blue_ai.scripts.train_agents import (
    categorize,
    categorize_pl,
    run_trial,
)

import pandas as pd
import polars as pl

from copy import deepcopy
from anytree import AnyNode, LevelGroupOrderIter, LevelOrderIter, RenderTree

from typing import List

import pickle


STEPS_PER_STAGE = [100_000, 100_000]
# STEPS_PER_STAGE = [s // 1000 for s in STEPS_PER_STAGE]


class NamedEnvironment(Image2VecWrapper):
    def __init__(self, env, name):
        super().__init__(env)
        self.name = name

    def __repr__(self):
        return f"{self.name} {hex(id(self))}"


def create_custom_tree(
    parent, depth, branches, current_depth=1, do_deepcopy=True, trial_id=None, path=""
):
    if current_depth > depth:
        return

    for branch in branches:
        branch = deepcopy(branch) if do_deepcopy else branch
        p = path + branch.name[0]

        child = AnyNode(parent=parent, env=branch, trial_id=trial_id, env_path=p)
        create_custom_tree(
            child, depth, branches, current_depth + 1, trial_id=trial_id, path=p
        )


def main():
    branches = [
        NamedEnvironment(
            TransientGoals(
                render_mode="none",
                transient_reward=0.5,
                n_transient_obstacles=0,
            ),
            name="Amazing",
        ),
        NamedEnvironment(
            TransientGoals(render_mode="none", transient_penalty=-0.2),
            name="Good",
        ),
        NamedEnvironment(
            TransientGoals(render_mode="none", transient_reward=0.1),
            name="Bad",
        ),
        NamedEnvironment(
            TransientGoals(
                render_mode="none",
                transient_reward=0.1,
                n_transient_obstacles=5,
                n_transient_goals=1,
            ),
            name="Hell",
        ),
    ]

    assert len(names := [b.name[0] for b in branches]) == len(
        set(names)
    ), "Ensure that all environments have a unique first char"

    agents: List[BaseAgent] = [
        HealthyAgent(),
        # SpineLossDepression(),
        RehabiliationAgent(),
    ]

    base = AnyNode()
    for i in range(20):
        agents = deepcopy(agents)
        for a in agents:
            agent_node = AnyNode(
                agent=a, parent=base, results=pl.DataFrame(), trial_id=i
            )
            create_custom_tree(agent_node, 2, branches, trial_id=i)

    total = sum(
        [
            STEPS_PER_STAGE[len(node.ancestors) - 2]
            for node in LevelOrderIter(base, filter_=lambda x: len(x.ancestors) >= 2)
        ]
    )
    tbar = tqdm(total=total, initial=0)

    # Iterate breadth first over the tree and agents, skipping the "base" node and the "agent nodes"
    # used for constructing the decision tree
    node: AnyNode
    for node in LevelOrderIter(base, filter_=lambda x: len(x.ancestors) >= 2):
        stage = len(node.ancestors) - 2

        node.agent = deepcopy(node.parent.agent)
        node.agent.state_change(stage=stage)

        starting_episode_num = 0
        starting_step = 0
        prequel = node.parent.results

        # Inherit initial data from parent
        if len(node.parent.results) > 0:
            starting_episode_num = node.parent.results["episode"].max()
            starting_step = node.parent.results["step"].max()

            # A section of data from the previous run, useful for carrying trend data across
            prequel = node.parent.results.filter(
                abs(pl.col("step") - STEPS_PER_STAGE[stage]) <= 5000
            )

        r, _, _ = run_trial(
            agent=node.agent,
            env=node.env,
            steps=STEPS_PER_STAGE[stage],
            tbar=tbar,
            trial_id=node.trial_id,
            # So that this data lines up bettwen the runs
            starting_episode_num=starting_episode_num + 1,
            starting_step=starting_step + 1,
            polars_ok=True,
        )
        r = r.with_columns(
            path=pl.lit(node.env_path),
            env_type=pl.lit(node.env.name),
        )

        node.results = pl.concat([prequel, r])

        node.results = node.results.with_columns(
            path=pl.lit(node.env_path),
            env_type=pl.lit(node.env.name),
        )

        (save_path := DATA_PATH / "branching_chunks").mkdir(exist_ok=True)

        file_name = (
            save_path
            / f"{node.agent.file_display_name()}_{node.env_path}_{node.trial_id}"
        )
        categorize_pl(node.results)
        node.results.write_parquet(file_name.with_suffix(".parquet"))

        if node.is_leaf:
            node.results.drop_in_place("layer")

            for a in node.ancestors:
                if hasattr(a, "results") and len(a.results) > 0:
                    a.results.drop_in_place("layer")

    frames = [node.results for node in base.descendants if len(node.results) > 0]

    combined_results = pl.concat(
        frames,
        how="diagonal",
    )

    # combined_results.drop_in_place("layer")

    combined_results = categorize_pl(combined_results)

    combined_results.write_parquet(DATA_PATH / "branching.parquet")
    # combined_results.write_csv(DATA_PATH / "branching.csv")


def get_len(df, col):
    try:
        return len(df[col].unique())
    except:
        return -1


if __name__ == "__main__":

    main()
