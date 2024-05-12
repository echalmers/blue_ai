from train_agents import run_trial
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.agents.agent_classes import BaseAgent, HealthyAgent, SpineLossDepression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from constants import DATA_PATH

pd.options.display.width = 0


results = []
for rep in range(10):
    for decay in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-1, 1e0]:
        # for agent in [HealthyAgent(), SpineLossDepression()]:
        env = Image2VecWrapper(
            TransientGoals(
                render_mode="none", transient_reward=0.25, termination_reward=1
            )
        )
        agent = BaseAgent(weight_decay=decay)

        df, agent, env = run_trial(agent, env, steps=20_000)
        results.append(
            {
                "decay": decay,
                # 'agent': agent.__class__.__name__,
                "rep": rep,
                "reward": df["reward"].values[-1000:].mean(),
            }
        )
results = pd.DataFrame(results)
results.to_csv(DATA_PATH / "decay_sweep.csv")
print(results)
# sns.barplot(results, hue='decay', y='reward')
sns.lineplot(results, x="decay", y="reward")
plt.xscale("log")
plt.show()
