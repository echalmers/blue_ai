from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.scripts.constants import DATA_PATH

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


import pickle
from pathlib import Path
import time
import sys

def view_reconstruction(directory: Path, trauma: bool):
    mode = 'interactive' # there is also datagen in the schizophrenia files

    with open(DATA_PATH / f'{directory.name}/interpretation_models{"_traumatized" if trauma else ""}.pkl', 'rb') as f:
        interpretation_models = pickle.load(f)
        interpretation_models['agent_name'] = interpretation_models['agent']
        interpretation_models['agent_name'] = interpretation_models['agent_name'].astype(str).replace(
            {'HealthyAgent': 'Healthy',
             'PTSDAgent': 'PTSD'
             }
        )
    
    def plot(state):
        for i in range(3):
            ax[i].cla()

        ax[0].imshow(env.render())
        ax[1].imshow(Image2VecWrapper.observation_to_image(state))
        state = torch.tensor(np.expand_dims(state, 0).astype(np.float32),
                                device=interpretation_models['agent'][0].device)

        for index, row in interpretation_models[interpretation_models['filename'].str.contains(f'_0{'_traumatized'if trauma else ''}.pkl')].iterrows():
            recon = row['interpretation_model'].get_reconstructions(observations=state)[1][0]
            mse = nn.MSELoss()(recon, state)
            print(row['filename'], mse)
            recon[recon < 0] = 0
            ax[2 + index].imshow(Image2VecWrapper.observation_to_image(recon.cpu() ** 1.5, closest=True))
            ax[2 + index].set_title(f"{row['agent_name']} reconstructed")  # ({round(float(mse), 2)})")

        for i in range(4):
            ax[i].set_xticks([])
            ax[i].set_yticks([])

        for i in range(1, 4):
            t = plt.Polygon([[1.75, 4.25], [2.25, 4.25], [2, 3.75]], color='red')
            ax[i].add_patch(t)

        ax[1].set_title('visual input')
        plt.pause(0.01)
    
    if mode == 'interactive':

        # create an environment
        env = Image2VecWrapper(
                TransientGoals(
                    render_mode="rgb_array", transient_reward=0.25, termination_reward=1,
                    #n_transient_obstacles=1, transient_penalty=-100, transient_obstacles=[[4,1]]
                    transient_locations=[[1,4],[4,2],[5,1]],
                    wall_locations =[[3,2],[3,3],[3,4],[3,5],[3,6]],
                    n_transient_obstacles=0
                )
            )
        state, _ = env.reset()

        def process(event):
            folder_path = directory / "img" / "reexperiences"
            folder_path.mkdir(parents=True, exist_ok=True)

            global state
            if event.key == 'left':
                action = 0
            elif event.key == 'right':
                action = 1
            elif event.key == 'up':
                action = 2
            elif event.key == 'z':
                plt.savefig(folder_path / f'{time.time()}.png', dpi=300)
                return
            else:
                return

            state, _, done, _, _ = env.step(action)
            if done:
                state, _ = env.reset()
            plot(state)


        # create figure window
        fig, ax = plt.subplots(1, 4, figsize=(10, 4))
        fig.canvas.mpl_connect('key_press_event', process)
        fig.suptitle(f'Reconstructions {'after' if trauma else 'before'} trauma')
        plot(state)

        plt.show()
    
if __name__ == "__main__":
    view_reconstruction(DATA_PATH / sys.argv[1], trauma=True)