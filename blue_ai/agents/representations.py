import numpy as np

from blue_ai.agents.dqn import DQN

import torch.nn as nn
import torch


class RepresentationProbe:

    def __init__(self, agent: DQN, memory_agent: DQN = None):
        self.agent = agent
        self.memory_agent = memory_agent or agent
        self._internal_activations = None

        # get sample input
        state = self.agent.transition_memory.states[0]

        # find and register the middle layer
        layers = [layer for layer in self.agent.policy_net if isinstance(layer, nn.Linear)]
        middle_layer = layers[len(layers)//2]
        middle_layer.register_forward_hook(self._capture_activations)

        # create model to reconstruct inputs from internal representations
        self.model = nn.Sequential(
            nn.Linear(middle_layer.in_features + middle_layer.out_features, state.numel()),
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
            observations, _, _, _, _ = self.memory_agent.transition_memory.sample(1000)
            self.agent.policy_net(observations)
            reconstruction = self.model(self._internal_activations)
            loss = loss_fn(reconstruction, observations)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return losses

    def get_reconstructions(self, observations):
        with torch.no_grad():
            self.agent.policy_net(observations)
            reconstruct = self.model(self._internal_activations)
        return observations, reconstruct

    def _capture_activations(self, layer, input, output):
        self._internal_activations = torch.hstack((input[0].detach(), output.detach()))


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
    depressed_probe = RepresentationProbe(depressed_agent, memory_agent=None)
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

        for i in range(4):
            ax[i].cla()

        ax[0].imshow(env.render())
        ax[1].imshow(Image2VecWrapper.observation_to_image(state))
        state = torch.tensor(np.expand_dims(state, 0).astype(np.float32), device=healthy_agent.device)

        healthy_recon = healthy_probe.get_reconstructions(observations=state)[1][0].cpu()
        depressed_recon = depressed_probe.get_reconstructions(observations=state)[1][0].cpu()
        healthy_recon[healthy_recon < 0] = 0
        # healthy_recon /= healthy_recon.max()
        depressed_recon[depressed_recon < 0] = 0
        # depressed_recon /= depressed_recon.max()

        ax[2].imshow(Image2VecWrapper.observation_to_image(healthy_recon ** 1.5))
        ax[3].imshow(Image2VecWrapper.observation_to_image(depressed_recon ** 1.5))

        for i in range(4):
            ax[i].set_xticks([])
            ax[i].set_yticks([])

        for i in range(1, 4):
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

        state, _, done, _, _ = env.step(action)
        if done:
            state, _ = env.reset()
        plot(state)

    # create figure window
    fig, ax = plt.subplots(1, 4)
    fig.canvas.mpl_connect('key_press_event', process)
    plot(state)

    plt.show()

