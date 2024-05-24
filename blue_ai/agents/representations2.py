import numpy as np

from blue_ai.agents.dqn import DQN

import torch.nn as nn
import torch


class RepresentationProbe:

    def __init__(self, agent: DQN):
        self.agent = agent
        self._internal_activations = None

        # get sample input
        state = self.agent.transition_memory.states[0]

        # find and register the middle layer
        layers = [layer for layer in self.agent.policy_net if isinstance(layer, nn.Linear)]
        middle_layer = layers[len(layers)//2]
        middle_layer.register_forward_hook(self._capture_activations)

        # create model to reconstruct inputs from internal representations
        self.model = nn.Sequential(
            nn.Linear(middle_layer.in_features, state.numel()),
            nn.Tanh(),
            nn.Linear(state.numel(), state.numel()),
            nn.Unflatten(1, state.shape)
        )
        print(self.model)
        self.model.to(self.agent.device)

    def fit(self):
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05, weight_decay=1e-6)

        # fit model
        losses = []
        for i in range(200):
            observations, _, _, _, _ = self.agent.transition_memory.sample(1000)
            self.agent.policy_net(observations)
            reconstruction = self.model(self._internal_activations)
            loss = loss_fn(reconstruction, observations)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return losses

    def sample_reconstructions(self, n=None, observations=None):
        with torch.no_grad():
            if n:
                observations, _, _, _, _ = self.agent.transition_memory.sample(n)
            self.agent.policy_net(observations)
            reconstruct = self.model(self._internal_activations)
        return observations, reconstruct

    def _capture_activations(self, layer, input, output):
        self._internal_activations = input[0].detach()


if __name__ == '__main__':
    from blue_ai.scripts.train_agents import load_trial
    from blue_ai.scripts.constants import DATA_PATH
    from blue_ai.envs.custom_wrappers import Image2VecWrapper
    from matplotlib import pyplot as plt
    from blue_ai.envs.transient_goals import TransientGoals

    # create models to interpret networks' hidden layers
    _, healthy_agent, _ = load_trial(DATA_PATH / 'HealthyAgent_0.pkl')
    _, depressed_agent, _ = load_trial(DATA_PATH / 'SpineLossDepression_0.pkl')

    healthy_probe = RepresentationProbe(healthy_agent)
    l = healthy_probe.fit()
    print(l[-1])
    # plt.plot(l)
    depressed_probe = RepresentationProbe(depressed_agent)
    l = depressed_probe.fit()
    print(l[-1])
    # plt.plot(l)

    # create an environment
    env = Image2VecWrapper(
        TransientGoals(
            render_mode="rgb_array", transient_reward=0.25, termination_reward=1
        )
    )
    state, _ = env.reset()

    def plot(state):
        ax[0].cla()
        ax[1].cla()
        ax[2].cla()
        print('plotting', state)
        ax[0].imshow(env.render())
        ax[1].imshow(Image2VecWrapper.observation_to_image(state))
        state = torch.Tensor(np.expand_dims(state, 0))
        ax[2].imshow(Image2VecWrapper.observation_to_image(healthy_probe.sample_reconstructions(observations=state)[1][0]))
        ax[3].imshow(Image2VecWrapper.observation_to_image(depressed_probe.sample_reconstructions(observations=state)[1][0]))

        for i in range(4):
            ax[i].set_xticks([])
            ax[i].set_yticks([])

        ax[1].set_title('visual input')
        ax[2].set_title('healthy reconstructed')
        ax[2].set_title('depressed reconstructed')
        plt.pause(0.1)

    def process(event):
        global state
        if event.key == 'left':
            action = 0
        elif event.key == 'right':
            action = 1
        elif event.key == 'up':
            action = 2

        state, _, done, _, _ = env.step(action)
        if done:
            state, _ = env.reset()
        plot(state)

    # create figure window
    fig, ax = plt.subplots(1, 4)
    fig.canvas.mpl_connect('key_press_event', process)
    plot(state)

    plt.show()

