import pandas as pd

from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.scripts.train_dqn import load_trial, Image2VecWrapper
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

plt.subplot(1,2,1)
env = Image2VecWrapper(TransientGoals(img_filename='env1.png', render_mode='rgb_array'))
state, _ = env.reset()
plt.imshow(env.render())
plt.xticks([])
plt.yticks([])
plt.subplot(1, 2, 2)
state, _ = env.reset()
plt.imshow(env.render())
plt.xticks([])
plt.yticks([])

fig, ax = plt.subplots(1, 2)

env = Image2VecWrapper(TransientGoals(img_filename='env1.png', transient_locations=[(5, 6)], agent_start_pos=(5, 5), agent_start_dir=0, render_mode='rgb_array'))
state, _ = env.reset()
ax[0].imshow(env.render())
ax[0].set_xticks([])
ax[0].set_yticks([])

values = []
for trial in range(10):
    results_healthy, agent_healthy, _ = load_trial(os.path.join('.', 'data', f'highterminal_0_{trial}_.pkl'))
    results_dep, agent_dep, _ = load_trial(os.path.join('.', 'data', f'highterminal_50_{trial}_.pkl'))

    healthy_values = agent_healthy.get_action_values(np.expand_dims(state, 0)).numpy()
    values.append([0, 'turn left', healthy_values[0]])
    values.append([0, 'turn right', healthy_values[1]])
    values.append([0, 'forward', healthy_values[2]])

    dep_values = agent_dep.get_action_values(np.expand_dims(state, 0)).numpy()
    values.append([50, 'turn left', dep_values[0]])
    values.append([50, 'turn right', dep_values[1]])
    values.append([50, 'forward', dep_values[2]])

values = pd.DataFrame(data=values, columns=['dropout (%)', 'action', 'value'])
print(values)

plt.sca(ax[1])
p = sns.barplot(data=values, x='dropout (%)', y='value', hue='action', color='grey', edgecolor=".5")
p.patches[0].set_facecolor('skyblue')
p.patches[2].set_facecolor('skyblue')
p.patches[4].set_facecolor('skyblue')
p.patches[1].set_facecolor('salmon')
p.patches[3].set_facecolor('salmon')
p.patches[5].set_facecolor('salmon')
plt.legend([],[], frameon=False)
plt.text(x=p.patches[0].get_center()[0], y=0.2, s='↶', fontsize='x-large', fontweight='heavy', ha='center')
plt.text(x=p.patches[2].get_center()[0], y=0.2, s='↑', fontsize='x-large', fontweight='heavy', ha='center')
plt.text(x=p.patches[4].get_center()[0], y=0.2, s='↷', fontsize='x-large', fontweight='heavy', ha='center')
plt.text(x=p.patches[1].get_center()[0], y=0.2, s='↶', fontsize='x-large', fontweight='heavy', ha='center')
plt.text(x=p.patches[3].get_center()[0], y=0.2, s='↑', fontsize='x-large', fontweight='heavy', ha='center')
plt.text(x=p.patches[5].get_center()[0], y=0.2, s='↷', fontsize='x-large', fontweight='heavy', ha='center')

plt.xlabel('')
plt.ylabel('value perceived by agent')
plt.xticks(ticks=[0, 1], labels=['0% dropout\nhealthy', '50% dropout\ndepressed'])

fig.suptitle('anhedonia')
plt.show()