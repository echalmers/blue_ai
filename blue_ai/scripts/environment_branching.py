from tqdm import tqdm
from blue_ai.agents.agent_classes import (
    BaseAgent,
    HealthyAgent,
    RehabiliationAgent,
    SpineLossDepression,
)
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH
from blue_ai.scripts.train_agents import (
    categorize_pl,
    run_trial,
)

import numpy as np

import polars as pl

from copy import deepcopy
from anytree import AnyNode, LevelOrderIter

from typing import List


STEPS_PER_STAGE = [5000, 10_000]


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
                render_mode="none", transient_penalty=-0.1, n_transient_goals=5
            ),
            name="Good",
        ),
        NamedEnvironment(
            TransientGoals(render_mode="none", n_transient_obstacles=5),
            name="Bad",
        ),
    ]

    assert len(names := [b.name[0] for b in branches]) == len(
        set(names)
    ), "Ensure that all environments have a unique first char"

    agents: List[BaseAgent] = [
        HealthyAgent(),
        SpineLossDepression(),
        RehabiliationAgent(),
    ]

    base = AnyNode()
    for i in range(5):
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

            transition_centers = np.concat([[0], np.cumsum(STEPS_PER_STAGE)])

            # A section of data from the previous run, useful for carrying trend
            # data across
            prequel = (node.parent.results).filter(
                abs(pl.col("step") - transition_centers[stage]) <= 1500
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

        categorize_pl(node.results)

    frames = [node.results for node in base.descendants if len(node.results) > 0]

    combined_results = pl.concat(
        frames,
        how="diagonal",
    )

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
