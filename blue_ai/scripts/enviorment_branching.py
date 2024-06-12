from tqdm import tqdm
from blue_ai.agents.agent_classes import (
    BaseAgent,
    HealthyAgent,
    RehabiliationAgent,
)
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH, N_TRIALS
from blue_ai.scripts.train_agents import run_trial

import pandas as pd

from copy import deepcopy
from anytree import AnyNode, LevelGroupOrderIter, LevelOrderIter, RenderTree

from typing import List

import pickle


STEPS_PER_STAGE = [40_000, 40_000]
# STEPS_PER_STAGE = [s // 100 for s in STEPS_PER_STAGE]


class NamedEnvironment(Image2VecWrapper):
    def __init__(self, env, name):
        super().__init__(env)
        self.name = name

    def __repr__(self):
        return f"{self.name} {hex(id(self))}"


def create_custom_tree(
    parent, depth, branches, current_depth=1, do_deepcopy=True, trial_id=None
):
    if current_depth > depth:
        return

    for branch in branches:
        branch = deepcopy(branch) if do_deepcopy else branch
        child = AnyNode(parent=parent, env=branch, trial_id=trial_id)
        create_custom_tree(child, depth, branches, current_depth + 1, trial_id=trial_id)


def main():
    branches = [
        NamedEnvironment(
            TransientGoals(
                render_mode="none", n_transient_goals=5, transient_penalty=-0.1
            ),
            name="Good",
        ),
        NamedEnvironment(
            TransientGoals(render_mode="none", n_transient_obstacles=5),
            name="Bad",
        ),
    ]

    agents: List[BaseAgent] = [
        # HealthyAgent(),
        # SpineLossDepression(),
        RehabiliationAgent(),
    ]

    base = AnyNode()
    for i in range(20):
        agents = deepcopy(agents)
        for a in agents:
            agent_node = AnyNode(
                agent=a, parent=base, results=pd.DataFrame(), trial_id=i
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
        node.results = deepcopy(node.parent.results)
        node.agent.state_change(stage=stage)

        # Inherit initial data from parent
        if len(node.parent.results) > 0:
            starting_episode_num = node.parent.results["episode"].max()
            starting_cumalative_reward = node.parent.results["cumulative_reward"].max()
            starting_step = node.parent.results["step"].max()
        else:
            starting_episode_num = 0
            starting_cumalative_reward = 0
            starting_step = 0

        r, _, _ = run_trial(
            node.agent,
            node.env,
            STEPS_PER_STAGE[stage],
            tbar=tbar,
            trial_id=node.trial_id,
            # So that this data lines up bettwen the runs
            starting_episode_num=starting_episode_num,
            starting_step=starting_step,
        )

        node.results = pd.concat([node.results, r])

        node.results["path"] = "".join(
            [a.env.name[0] for a in (node,) + node.ancestors if len(a.ancestors) >= 2]
        )

        node.results["env_type"] = node.env.name
        node.results["starting_cumalative_reward"] = starting_cumalative_reward

    with open(DATA_PATH / "full_tree.pkl", "wb") as f:
        pickle.dump(base, f)

    combined_results = pd.concat([node.results for node in base.leaves])

    with open(DATA_PATH / "branching.pkl", "wb") as f:
        pickle.dump(combined_results, f)


if __name__ == "__main__":
    main()
