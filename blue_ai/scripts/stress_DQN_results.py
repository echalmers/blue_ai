import pandas as pd
from constants import DATA_PATH
import matplotlib.pyplot as plt
import seaborn as sns
from blue_ai.envs.transient_goals import TransientGoals
from matplotlib.patches import ConnectionPatch


results = pd.read_csv(DATA_PATH / 'stress.csv', index_col=None)

mosaic = """
abc
ddd
eee
"""
fig, axes = plt.subplot_mosaic(mosaic, figsize=(9, 5))

good_env = TransientGoals(render_mode="rgb_array", transient_reward=0.25, termination_reward=1)  #, transient_locations=[(3, 3), (5, 4)]) )
bad_env = TransientGoals(render_mode="rgb_array", transient_reward=0.25, termination_reward=1, n_transient_obstacles=10, n_transient_goals=0)
good_env.reset()
bad_env.reset()
axes['a'].imshow(good_env.render())
axes['b'].imshow(bad_env.render())
axes['c'].imshow(good_env.render())
for axis in ['a', 'b', 'c']:
    axes[axis].set_xticks([])
    axes[axis].set_yticks([])

plt.sca(axes['d'])
sns.lineplot(results[results['stress_length'] == 20], x='episode', y='expected_shortterm')
# sns.lineplot(results[results['stress_length'] == 20], x='episode', y='expected_longterm')
axes['d'].axvspan(400, 420, color='red', alpha=0.25)
# axes['d'].set_xlim((0, 1050))
axes['d'].set_xticks([])
axes['d'].set_xlabel('')
axes['d'].set_ylabel('average\nreward')
axes['d'].set_title('brief stress period')

plt.sca(axes['e'])
sns.lineplot(results[results['stress_length'] == 50], x='episode', y='expected_shortterm')
# sns.lineplot(results[results['stress_length'] == 50], x='episode', y='expected_longterm')
axes['e'].axvspan(400, 450, color='red', alpha=0.25)
# axes['e'].set_xlim((0, 1050))
axes['e'].set_ylabel('average\nreward')
axes['e'].set_title('prolonged stress period')

# con = ConnectionPatch(xyA=(0, 0), xyB=(0, 0), coordsA='data', coordsB='data', axesA=axes['a'], axesB=axes['d'])
# axes['d'].add_artist(con)
plt.tight_layout()
plt.show()


