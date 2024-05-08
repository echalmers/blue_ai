from train_agents import run_trial
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.agents.agent_classes import BaseAgent, HealthyAgent, SpineLossDepression
import pandas as pd

pd.options.display.width = 0
import seaborn as sns
import matplotlib.pyplot as plt
import os


results = []
for rep in range(10):
    for agent in [HealthyAgent(), SpineLossDepression()]:
        for env in [
            Image2VecWrapper(TransientGoals(render_mode="none")),
            Image2VecWrapper(
                TransientGoals(
                    render_mode="none", n_transient_goals=0, n_transient_obstacles=0
                )
            ),
        ]:

            df, agent, env = run_trial(agent, env, steps=20_000)
            results.append(
                {
                    "env": "simple" if env.n_transient_goals == 0 else "complex",
                    "agent": agent.__class__.__name__,
                    "rep": rep,
                    "reward": df["reward"].values[-1000:].mean(),
                }
            )
results = pd.DataFrame(results)
results.to_csv(os.path.join(".", "data", "simple_env_comparison.csv"))
print(results)
sns.catplot(data=results, kind="bar", x="env", y="reward", hue="agent")
plt.show()
