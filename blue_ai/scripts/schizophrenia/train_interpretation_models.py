from blue_ai.scripts.train_agents import load_trial
from blue_ai.agents.dqn import DQN
from blue_ai.scripts.constants import DATA_PATH

import pandas as pd
import torch.nn as nn
import torch
import pickle


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

    files = ['HealthyAgent_0.pkl', 'SpineLossDepression_0.pkl', 'SchizophrenicAgent_0.pkl']

    interpretation_models = pd.DataFrame({
        'filename': files,
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