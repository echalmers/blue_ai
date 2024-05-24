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

    _, healthy_agent, _ = load_trial(DATA_PATH / 'HealthyAgent_0.pkl')
    _, depressed_agent, _ = load_trial(DATA_PATH / 'SpineLossDepression_0.pkl')

    healthy_probe = RepresentationProbe(healthy_agent)
    l = healthy_probe.fit()
    print(l[-1])
    plt.plot(l)
    depressed_probe = RepresentationProbe(depressed_agent)
    l = depressed_probe.fit()
    print(l[-1])
    plt.plot(l)

    n = 6
    fig, ax = plt.subplots(n, 4)
    obs, healthy_reconstructions = healthy_probe.sample_reconstructions(n=n)
    _, depressed_reconstructions = depressed_probe.sample_reconstructions(observations=obs)
    for i in range(n):
        this_obs = obs[i].cpu().numpy()
        healthy_recon = healthy_reconstructions[i].cpu().numpy()
        depressed_recon = depressed_reconstructions[i].cpu().numpy()

        ax[i, 0].plot(this_obs.flatten(), 'k')
        ax[i, 0].plot(healthy_recon.flatten(), 'b')
        ax[i, 0].plot(depressed_recon.flatten(), 'r')
        if i == 0:
            ax[i, 0].legend(['obs', 'healthy', 'depressed'])

        ax[i, 1].imshow(Image2VecWrapper.observation_to_image(this_obs))

        healthy_recon[healthy_recon < 0] = 0
        healthy_recon /= healthy_recon.max()
        depressed_recon[depressed_recon < 0] = 0
        depressed_recon /= depressed_recon.max()

        ax[i, 2].imshow(Image2VecWrapper.observation_to_image(healthy_recon ** 2))
        ax[i, 3].imshow(Image2VecWrapper.observation_to_image(depressed_recon ** 1.5))
    plt.show()
