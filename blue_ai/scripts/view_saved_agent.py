from blue_ai.scripts.train_dqn import load_trial, Image2VecWrapper
from blue_ai.envs.transient_goals import TransientGoals

import os
import numpy as np


filename = os.path.join('.', 'data', '0.25_1_0-0.pkl')
_, agent = load_trial(filename)

# create the environment
env = Image2VecWrapper(TransientGoals(tile_size=32, render_mode='human'))
state, _ = env.reset()

steps_this_episode = 0
for step in range(1000):

    # get & execute action
    action = agent.select_action(np.expand_dims(state, 0))
    new_state, reward, done, _, _ = env.step(action)

    # use this experience to update agent
    agent.update(state, action, reward, new_state, done=False)

    # reset environment if done (ideally env would do this itself)
    if done or steps_this_episode > 500:
        state, _ = env.reset()
        steps_this_episode = 0
    else:
        state = new_state
