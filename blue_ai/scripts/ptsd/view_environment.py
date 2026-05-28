from blue_ai.scripts.train_agents import load_dataset
import pandas as pd
pd.options.display.width = None
import numpy as np
import matplotlib.pyplot as plt
from blue_ai.envs.transient_goals import TransientGoals


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
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.show()
