from blue_ai.scripts.train_agents import run_trial, save_trial
from blue_ai.agents.agent_classes import HealthyAgent
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import trange


def main():
    for rep in trange(10):
        results1, agent, env = run_trial(
            trial_id=rep,
            agent=HealthyAgent(),
            env=Image2VecWrapper(TransientGoals(render_mode="none")),
            steps=40_000,
        )

        agent.optimizer = torch.optim.Adam(
            agent.policy_net.parameters(), lr=agent.lr, weight_decay=3e-3
        )

        results2, agent, env = run_trial(
            trial_id=rep,
            agent=agent,
            env=env,
            steps=40_000,
        )

        agent.optimizer = torch.optim.Adam(
            agent.policy_net.parameters(), lr=agent.lr, weight_decay=0
        )

        results3, agent, env = run_trial(
            trial_id=rep,
            agent=agent,
            env=env,
            steps=30_000,
        )

        results = pd.concat([results1, results2, results3], ignore_index=True)
        results["cumulative_reward"] = results["reward"].cumsum()
        results["step"] = np.arange(results.shape[0])
        save_trial(
            results,
            agent,
            env,
            DATA_PATH / f"rehabilitate_{rep}.pkl",
        )

    plt.plot(results["cumulative_reward"])
    plt.show()


if __name__ == "__main__":
    main()
