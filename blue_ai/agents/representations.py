import numpy as np

from blue_ai.agents.dqn import DQN

import torch.nn as nn
import torch


class RepresentationProbe:

    def __init__(self, agent: DQN, memory_agent: DQN = None):
        self.agent = agent
        self.memory_agent = memory_agent or agent
        self._internal_activations = dict()

        # get sample input
        state = self.agent.transition_memory.states[0]

        # find and register the middle layer
        layers = [layer for layer in self.agent.policy_net if isinstance(layer, nn.Linear)]
        num_internal_units = 0
        for layer in layers[1:]:
            layer.register_forward_hook(self._capture_activations)
            num_internal_units += layer.in_features

        # create model to reconstruct inputs from internal representations
        self.model = nn.Sequential(
            nn.Linear(num_internal_units, state.numel()*2),
            nn.Tanh(),
            nn.Linear(state.numel()*2, state.numel()*2),
            nn.Tanh(),
            # nn.Linear(state.numel() * 2, state.numel() * 2),
            # nn.Tanh(),
            nn.Linear(state.numel()*2, state.numel()),
            nn.Unflatten(1, state.shape)
        )
        print(self.model)
        self.model.to(self.agent.device)

    def fit(self):
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-6)

        # fit model
        losses = []
        for i in range(5_000):
            observations, _, _, _, _ = self.memory_agent.transition_memory.sample(1000)
            with torch.no_grad():
                self.agent.policy_net(observations)
            reconstruction = self.model(torch.hstack(list(self._internal_activations.values())))
            loss = loss_fn(reconstruction, observations)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return losses

    def get_reconstructions(self, observations):
        with torch.no_grad():
            self.agent.policy_net(observations)
            reconstruct = self.model(torch.hstack(list(self._internal_activations.values())))
        return observations, reconstruct

    def _capture_activations(self, layer, input, output):
        self._internal_activations[layer] = input[0]


if __name__ == '__main__':
    from blue_ai.scripts.train_agents import load_trial
    from blue_ai.scripts.constants import DATA_PATH
    from blue_ai.envs.custom_wrappers import Image2VecWrapper
    from matplotlib import pyplot as plt
    from blue_ai.envs.transient_goals import TransientGoals
    import pandas as pd
    import pickle
    import os


    if os.path.exists(DATA_PATH / 'interpretation_models.pkl'):
        with open(DATA_PATH / 'interpretation_models.pkl', 'rb') as f:
            interpretation_models = pickle.load(f)
    else:
        interpretation_models = pd.DataFrame({
            'filename': ['HealthyAgent_0.pkl', 'SpineLossDepression_0.pkl', 'PositiveLossAgent_0.pkl'],
            'agent': None,
            'interpretation_model': None,
        })

        for index, row in interpretation_models.iterrows():
            results, agent, _ = load_trial(DATA_PATH / row['filename'])
            probe = RepresentationProbe(agent)
            l = probe.fit()
            print(l[-1])
            row['interpretation_model'] = probe
            row['agent'] = agent
        with open(DATA_PATH / 'interpretation_models.pkl', 'wb') as f:
            pickle.dump(interpretation_models, f)

    # create an environment
    env = Image2VecWrapper(
        TransientGoals(
            render_mode="rgb_array", transient_reward=0.25, termination_reward=1
        ),
        noise_level=0.0
    )
    state, _ = env.reset()

    # add noise to the networks
    for index, row in interpretation_models.iterrows():
        noise_layer = row['interpretation_model'].agent.policy_net[1]
        assert hasattr(noise_layer, 'std')
        noise_layer.std = 0.15

    def plot(state):

        for i in range(5):
            ax[i].cla()

        ax[0].imshow(env.render())
        ax[1].imshow(Image2VecWrapper.observation_to_image(state))
        state = torch.tensor(np.expand_dims(state, 0).astype(np.float32), device=interpretation_models['agent'][0].device)

        for index, row in interpretation_models.iterrows():
            recon = row['interpretation_model'].get_reconstructions(observations=state)[1][0].cpu()
            print(row['filename'], nn.MSELoss()(recon, state))
            recon[recon < 0] = 0
            ax[2 + index].imshow(Image2VecWrapper.observation_to_image(recon ** 1.5))

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

