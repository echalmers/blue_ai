from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.scripts.train_dqn import load_trial, Image2VecWrapper
import numpy as np
import os


env = Image2VecWrapper(TransientGoals(img_filename='env1.png', transient_locations=[(5, 6)], agent_start_pos=(5, 5), agent_start_dir=0, render_mode='human'))
state, _ = env.reset()

results_healthy, agent_healthy = load_trial(os.path.join('.', 'data', '0-1.pkl'))
results_dep, agent_dep = load_trial(os.path.join('.', 'data', '50-0.pkl'))

print(agent_healthy.get_action_values(np.expand_dims(state, 0)))
print(agent_dep.get_action_values(np.expand_dims(state, 0)))
input('enter to quit')