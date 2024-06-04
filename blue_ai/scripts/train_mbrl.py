from blue_ai.scripts.train_agents import run_trial
from blue_ai.agents.tabular.mbrl import MBRL
from blue_ai.envs.custom_wrappers import Image2FlatVecWrapper
from blue_ai.envs.transient_goals import TransientGoals

from blue_ai.scripts.constants import DATA_PATH
import os

agent_save_path = DATA_PATH / "mbrl.pkl"

if os.path.exists(agent_save_path):
    agent = MBRL.load(agent_save_path, actions=[0, 1, 2])
else:
    agent = MBRL(
        actions=[0, 1, 2],
        max_value_iterations=1000,
        theta_threshold=0.001,
        epsilon=0.25,
        q_default=1,
    )

agent.max_value_iterations = 100
env = Image2FlatVecWrapper(
    TransientGoals(render_mode="human", transient_reward=0.25, termination_reward=1)
)  # set render mode to "human" to see the agent moving around

for i in range(10_000 // 500):
    results, agent, env = run_trial(agent=agent, env=env, steps=500)
    agent.save(agent_save_path)

print(results)
