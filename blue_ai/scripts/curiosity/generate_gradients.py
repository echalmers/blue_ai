import pandas as pd

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.envs.custom_wrappers import AbsolutePositionWrapper
from blue_ai.agents.mbrl import MBRL
import numpy as np
from matplotlib import pyplot as plt


# create the environment
env = AbsolutePositionWrapper(TransientGoals(
    render_mode='none', n_transient_obstacles=0, n_transient_goals=0, library=(6,1), goal=[(6, 6)])
)
state, _ = env.reset()

# create the agent
agent = MBRL(actions=[0,1,2], discount_factor=0.99, q_default=1)


STEPS = 10_000
steps_this_episode = 0
for step in range(STEPS):
    print(step)
    steps_this_episode += 1

    # get & execute action
    action = agent.select_action(state)
    new_state, reward, done, _, _ = env.step(action)

    # update the agent
    agent.update(state=state, new_state=new_state, reward=reward, done=done, action=action)

    # reset the environment if goal reached
    if done or steps_this_episode > 500:
        state, _ = env.reset()
        steps_this_episode = 0
    else:
        state = new_state


max_values = []
for state, action_values in agent.Q.table.items():
    max_values.append([state[1:], max(action_values.values())])
max_values = pd.DataFrame(max_values, columns=['state', 'max_value'])
max_values = max_values.groupby('state').max().reset_index()

image = np.zeros((8, 8))
for index, row in max_values.iterrows():
    image[row['state']] = row['max_value']
plt.imshow(image)
plt.colorbar()
plt.show()

agent.save(r'C:\Users\echalmers\PycharmProjects\blue_ai\blue_ai\scripts\data\gradient_6_6.pkl')