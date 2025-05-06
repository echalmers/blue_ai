import pandas as pd
import time

from blue_ai.envs.custom_wrappers import Image2VecWrapper
from blue_ai.envs.transient_goals import TransientGoals
from blue_ai.scripts.constants import DATA_PATH, _CURRENT_DIR
from blue_ai.scripts.schizophrenia.train_interpretation_models import RepresentationProbe

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn
import torch


if __name__ == '__main__':

    mode = 'interactive'  # interactive or datagen
    include = ['Healthy', 'Schizophrenic']  #, 'Depressed']

    with open(DATA_PATH / 'interpretation_models.pkl', 'rb') as f:
        interpretation_models = pickle.load(f)
        interpretation_models['agent_name'] = interpretation_models['agent']
        interpretation_models['agent_name'] = interpretation_models['agent_name'].astype(str).replace(
            {'HealthyAgent': 'Healthy',
             'SpineLossDepression': 'Depressed',
             'SchizophrenicAgent': 'Schizophrenic'
             }
        )
        interpretation_models = interpretation_models[interpretation_models['agent_name'].isin(include)].reset_index()


    def plot(state):

        for i in range(3):
            ax[i].cla()

        ax[0].imshow(env.render())
        ax[1].imshow(Image2VecWrapper.observation_to_image(state))
        state = torch.tensor(np.expand_dims(state, 0).astype(np.float32),
                             device=interpretation_models['agent'][0].device)

        for index, row in interpretation_models[interpretation_models['filename'].str.contains('_0.pkl')].iterrows():
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

    def add_noise(std):
        # add noise to the networks
        for index, row in interpretation_models.iterrows():
            for layer in row['interpretation_model'].agent.policy_net:
                if hasattr(layer, 'std'):
                    print(f'changing noise std from {layer.std} to {std}')
                    layer.std = std

    if mode == 'interactive':
        # create an environment
        env = Image2VecWrapper(
            TransientGoals(
                render_mode="rgb_array", transient_reward=0.25, termination_reward=1
            ),
            noise_level=0.0
        )
        state, _ = env.reset()

        add_noise(0.2)

        def process(event):
            global state
            if event.key == 'left':
                action = 0
            elif event.key == 'right':
                action = 1
            elif event.key == 'up':
                action = 2
            elif event.key == 'f':
                plt.savefig(DATA_PATH / 'hallucinations' / f'{time.time()}.png', dpi=300)
                return
            else:
                return

            state, _, done, _, _ = env.step(action)
            if done:
                state, _ = env.reset()
            plot(state)


        # create figure window
        fig, ax = plt.subplots(1, 4)
        fig.canvas.mpl_connect('key_press_event', process)
        plot(state)

        plt.show()

    elif mode == 'datagen':
        datapoints = []
        # env.env.render_mode = "human"

        for std in [0.45, 0.3, 0.15, 0]:
            add_noise(std)

            for index, row in interpretation_models.iterrows():
                print(std, row['filename'])
                for step in range(1_000):
                    # action = np.random.choice([0, 1, 2], p=[0.2, 0.2, 0.6])
                    # state, _, done, _, _ = env.step(action)
                    # if done:
                    #     state, _ = env.reset()
                    # create an environment
                    env = Image2VecWrapper(
                        TransientGoals(
                            render_mode="rgb_array", transient_reward=0.25, termination_reward=1,
                            agent_start_dir=np.random.randint(0, 4),
                            agent_start_pos=np.random.randint(1, 6, 2)
                        ),
                        noise_level=0.0
                    )
                    state, _ = env.reset()

                    state = torch.tensor(np.expand_dims(state, 0).astype(np.float32), device=row['agent'].device)
                    recon = row['interpretation_model'].get_reconstructions(observations=state)[1][0]
                    mse = nn.MSELoss()(recon, state).item()
                    itemwise_error = (recon - state.squeeze()).sum(axis=1).sum(axis=1).numpy()
                    datapoints.append(
                        {
                            'agent': row['agent_name'],
                            'std': std,
                            'mse': mse,
                            'error_wall': itemwise_error[0],
                            'error_goal': itemwise_error[1],
                            'error_goalNoTerminate': itemwise_error[2],
                            'error_obstacleNoTerminate': itemwise_error[3],
                        }
                    )

        with open(DATA_PATH / 'reconstruction_error.pkl', 'wb') as f:
            pickle.dump(datapoints, f)

        datapoints = pd.DataFrame(datapoints)
        # fig, axes = plt.subplots(1, 2)

        # plt.sca(axes[0])
        sns.lineplot(data=datapoints, x='std', y='mse', hue='agent', palette=['blue', 'orange'])
        plt.ylabel('mean-squared error in reconstructing visual input')
        plt.xlabel('standard deviation of added noise')

        # plt.sca(axes[1])
        # sns.lineplot(data=datapoints, x='std', y='error_wall', color='grey', style='agent')
        # sns.lineplot(data=datapoints, x='std', y='error_goal', color='green', style='agent')
        # sns.lineplot(data=datapoints, x='std', y='error_goalNoTerminate', color='blue', style='agent')
        # sns.lineplot(data=datapoints, x='std', y='error_obstacleNoTerminate', color='red', style='agent')
        plt.grid()


        datapoints.to_csv(_CURRENT_DIR / 'schizophrenia' / 'img' / 'recon_error_data.csv', index=None)
        plt.show()

    else:
        raise Exception('mode must be "interactive" or "datagen"')