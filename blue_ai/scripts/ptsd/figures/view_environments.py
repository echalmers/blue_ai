from blue_ai.scripts.train_agents import load_dataset
import pandas as pd
pd.options.display.width = None
import numpy as np
import matplotlib.pyplot as plt
from blue_ai.envs.transient_goals import TransientGoals


f, ax = plt.subplots(1, 2)  # , figsize=(12, 4))

env = TransientGoals(
    render_mode="rgb_array",
    n_transient_obstacles=1, n_transient_goals=2,
    agent_start_dir=1,
    # agent_start_pos=np.array([4, 2]),
    transient_locations=[[2, 4], [5, 1]],
    transient_obstacles=[[1, 6]],
    wall_locations =[[3,2],[3,3],[3,4],[3,5]],
    highlight=True
)
env.reset()

img = env.render()
ax[0].imshow(img)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title('typical episode')


# trauma env

env = TransientGoals(
    render_mode="rgb_array",
    n_transient_obstacles=1, n_transient_goals=0,
    agent_start_dir=0,
    agent_start_pos=np.array([3, 1]),
    # transient_locations=[[2, 4], [5, 1]],
    transient_obstacles=[[4, 1]],
    wall_locations =[[3,2],[3,3],[3,4],[3,5]],
    highlight=True
)
env.reset()

img = env.render()
ax[1].imshow(img)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title('"traumatic" episode')


plt.show()
