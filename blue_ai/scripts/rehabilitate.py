from blue_ai.agents.dqn import SpinelossLayer
from blue_ai.scripts.train_agents import run_trial, save_trial
from blue_ai.agents.agent_classes import HealthyAgent, WeightDropAgent
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.scripts.constants import DATA_PATH

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import trange


def main():
    for rep in trange(1):

        results1, agent, env = run_trial(
            trial_id=rep,
            agent=WeightDropAgent(),
            env=Image2VecWrapper(TransientGoals(render_mode="none")),
            steps=60_000,
            filename="test_healthy"
        )

        # agent.optimizer = torch.optim.Adam(
        #     agent.policy_net.parameters(), lr=agent.lr, weight_decay=3e-3
        # )

        dropout = 0.3


        # #depression on-set
        for layer in agent.policy_net.children():
            if isinstance(layer, SpinelossLayer):
                layer.set_dropout_rate(dropout)

        results2, agent, env = run_trial(
            trial_id=rep,
            agent=agent,
            env=env,
            steps=60_000,
            filename="test_depressed"

        )

        results = pd.concat([results1, results2], ignore_index=True)

        min_noise = .4
        max_noise = 1.0
        dropout_removal = True

        for layer in agent.policy_net.children():
            if isinstance(layer, SpinelossLayer):
                layer.set_noise_scale(min_noise, max_noise)
                layer.add_noise()

        results3, agent, env = run_trial(
            trial_id=rep,
            agent=agent,
            env=env,
            steps=20_000,
            filename="test_treated"
        )
        results = pd.concat([results, results3], ignore_index=True)


        for i in range(6):
            if dropout > 0:
                dropout -= .1
            else:
                dropout = 0.0

            for layer in agent.policy_net.children():
                if isinstance(layer, SpinelossLayer) & dropout_removal:
                    layer.set_dropout_rate(dropout)
                    print(dropout)

            results3, agent, env = run_trial(
                trial_id=rep,
                agent=agent,
                env=env,
                steps=10_000,
                filename="test_treated"
            )
            results = pd.concat([results, results3], ignore_index=True)

        results["cumulative_reward"] = results["reward"].cumsum()
        results['rolling_avg_reward'] = results['reward'].rolling(window=8000).mean()
        results["step"] = np.arange(results.shape[0])

        save_trial(
            results,
            agent,
            env,
            DATA_PATH / f"rehab_long_depression_{rep}.pkl",
            )

    plt.plot(results["rolling_avg_reward"])
    plt.suptitle("rolling average of reward per step")
    plt.title(f"dropout: {dropout} and  noise scale {min_noise} - {max_noise}, dropout removal:{dropout_removal}")

    plt.show()


if __name__ == "__main__":
    main()
