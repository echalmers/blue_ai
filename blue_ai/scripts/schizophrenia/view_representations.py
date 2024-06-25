import pandas as pd

from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.scripts.constants import DATA_PATH
from blue_ai.scripts.schizophrenia.train_interpretation_models import RepresentationProbe

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn
import torch


if __name__ == '__main__':

    mode = 'interactive'  # interactive or datagen

    with open(DATA_PATH / 'interpretation_models.pkl', 'rb') as f:
        interpretation_models = pickle.load(f)

    # create an environment
    env = Image2VecWrapper(
        TransientGoals(
            render_mode="rgb_array", transient_reward=0.25, termination_reward=1
        ),
        noise_level=0.0
    )
    state, _ = env.reset()

    def add_noise(std):
        # add noise to the networks
        for index, row in interpretation_models.iterrows():
            for layer in row['interpretation_model'].agent.policy_net:
                if hasattr(layer, 'std'):
                    print(f'changing noise std from {layer.std} to {std}')
                    layer.std = std

    if mode == 'interactive':
        add_noise(0.25)

        def plot(state):

            for i in range(5):
                ax[i].cla()

            ax[0].imshow(env.render())
            ax[1].imshow(Image2VecWrapper.observation_to_image(state))
            state = torch.tensor(np.expand_dims(state, 0).astype(np.float32),
                                 device=interpretation_models['agent'][0].device)

            for index, row in interpretation_models[interpretation_models['filename'].str.contains('_0.pkl')].iterrows():
                recon = row['interpretation_model'].get_reconstructions(observations=state)[1][0]
                print(row['filename'], nn.MSELoss()(recon, state))
                recon[recon < 0] = 0
                ax[2 + index].imshow(Image2VecWrapper.observation_to_image(recon.cpu() ** 1.5))

            for i in range(5):
                ax[i].set_xticks([])
                ax[i].set_yticks([])

            for i in range(1, 5):
                t = plt.Polygon([[1.75, 4.25], [2.25, 4.25], [2, 3.75]], color='red')
                ax[i].add_patch(t)

            ax[1].set_title('visual input')
            ax[2].set_title('healthy reconstructed')
            ax[3].set_title('depressed reconstructed')
            plt.pause(0.01)

        def process(event):
            global state
            if event.key == 'left':
                action = 0
            elif event.key == 'right':
                action = 1
            elif event.key == 'up':
                action = 2
            else:
                return

            state, _, done, _, _ = env.step(action)
            if done:
                state, _ = env.reset()
            plot(state)


        # create figure window
        fig, ax = plt.subplots(1, 5)
        fig.canvas.mpl_connect('key_press_event', process)
        plot(state)

        plt.show()

    elif mode == 'datagen':
        datapoints = []
        # env.env.render_mode = "human"

        for std in [0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            add_noise(std)

            for index, row in interpretation_models.iterrows():
                print(std, row['filename'])
                for step in range(100):
                    action = np.random.choice([0, 1, 2], p=[0.2, 0.2, 0.6])
                    state, _, done, _, _ = env.step(action)
                    if done:
                        state, _ = env.reset()

                    state = torch.tensor(np.expand_dims(state, 0).astype(np.float32), device=row['agent'].device)
                    recon = row['interpretation_model'].get_reconstructions(observations=state)[1][0]
                    datapoints.append(
                        {
                            'agent': row['agent'],
                            'std': std,
                            'mse': nn.MSELoss()(recon, state).item()
                        }
                    )

        with open(DATA_PATH / 'reconstruction_error.pkl', 'wb') as f:
            pickle.dump(datapoints, f)

        sns.lineplot(data=pd.DataFrame(datapoints), x='std', y='mse', hue='agent')
        plt.show()

    else:
        raise Exception('mode must be "interactive" or "datagen"')