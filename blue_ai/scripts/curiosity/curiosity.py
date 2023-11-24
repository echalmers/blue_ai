from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.agents.agent_classes import HealthyAgent
import numpy as np


# create the environment
env = Image2VecWrapper(TransientGoals(
    render_mode='human', n_transient_obstacles=0, n_transient_goals=0, library=(6,1), goal=[(1, 6), (6, 6)])
)
state, _ = env.reset()

# create the agent
agent = HealthyAgent()

# training loop
STEPS = 30_000
steps_this_episode = 0
for step in range(STEPS):
    print(step)
    steps_this_episode += 1

    # get & execute action
    action = agent.select_action(np.expand_dims(state, 0))
    new_state, reward, done, _, _ = env.step(action)

    # update the agent
    agent.update(state=state, new_state=new_state, reward=reward, done=done, action=action)

    # reset the environment if goal reached
    if done or steps_this_episode > 500:
        state, _ = env.reset()
        steps_this_episode = 0
    else:
        state = new_state