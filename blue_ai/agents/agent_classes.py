from blue_ai.agents.dqn import DQN, SpinelossLayer
from torch import nn
import numpy as np
import torch


from blue_ai.envs.custom_decay import PositivePenaltyLoss


class BaseAgent(DQN):

    def file_display_name(self):
        """
        Used for getting the name of used for saving the files, override this
        if you need to embedded custom information in the filename
        """
        return self.__class__.__name__

    def __init__(
        self,
        network: nn.Sequential | None = None,
        input_shape=(4, 5, 5),
        replay_buffer_size=10000,
        update_frequency=5,
        lr=0.01,
        sync_frequency=25,
        gamma=0.9,
        epsilon=0.05,
        batch_size=1500,
        weight_decay=0.0,
        softmax_temperature=None,
        loss_fn: torch.nn.Module | None = None,
    ):

        super().__init__(
            network=(
                network
                or nn.Sequential(
                    nn.Flatten(1, -1), nn.Linear(100, 10), nn.Sigmoid(), nn.Linear(10, 3)
                )
            ),
            input_shape=input_shape,
            replay_buffer_size=replay_buffer_size,
            update_frequency=update_frequency,
            lr=lr,
            sync_frequency=sync_frequency,
            gamma=gamma,
            epsilon=epsilon,
            softmax_temp=softmax_temperature,
            batch_size=batch_size,
            weight_decay=weight_decay,
            loss_fn=loss_fn,
        )

    def __repr__(self) -> str:
        return self.file_display_name()


class HealthyAgent(BaseAgent):

    display_name = "healthy"

    def __init__(self):
        super().__init__(weight_decay=0)


class SpineLossDepression(BaseAgent):

    display_name = "simulated spine loss"

    def __init__(self):
        super().__init__(weight_decay=1e-3)


class ContextDependentLearningRate(BaseAgent):

    display_name = "context-dependent learning rate"
    positive_scale = 0.5
    negative_scale = 2

    def update(self, state, action, reward, new_state, done):
        if reward > 0:
            reward *= self.positive_scale
        elif reward < 0:
            reward *= self.negative_scale

        super().update(state, action, reward, new_state, done)


class HighDiscountRate(BaseAgent):

    display_name = "high discounting"

    def __init__(self):
        super().__init__(weight_decay=0, gamma=0.5)


class HighExploration(BaseAgent):

    display_name = "high exploration"

    def __init__(self):
        super().__init__(softmax_temperature=1)


class ScaledTargets(BaseAgent):

    display_name = "scaled target value"

    def __init__(self):
        super().__init__(lr=0.01)


class ShiftedTargets(BaseAgent):
    display_name = "shifted target value"
    offset = 1

    def update(self, state, action, reward, new_state, done):

        self.transition_memory.add(state, action, reward, new_state, done)
        self.update_counter += 1
        if self.update_counter % self.update_frequency == 0:

            # sync value and policy networks
            self.sync_counter += 1
            if self.sync_counter % self.sync_frequency == 0:
                self.value_net.load_state_dict(self.policy_net.state_dict())

            s, a, r, ns, d = self.transition_memory.sample(self.batch_size)

            # get policy network's current value estimates
            state_action_values = self.policy_net(s)

            # get target value estimates, based on actual rewards and value net's predictions of next-state value
            with torch.no_grad():
                new_state_value, _ = self.value_net(ns).max(1)
            target_action_value = (
                r + self.gamma * new_state_value * (1 - d)
            ) - self.offset
            target_values = state_action_values.clone().detach()
            target_values[np.arange(target_values.shape[0]), a] = target_action_value

            # optimize loss
            loss = self.loss_fn(state_action_values, target_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()


class PositiveLossAgent(BaseAgent):
    display_name = "Positive Loss Agent"

    def __init__(self, alpha=1e-1, embed_alpha_in_filename=False):
        self.embed_alpha_in_filename = embed_alpha_in_filename
        self.alpha = alpha
        custom_loss_function = PositivePenaltyLoss(alpha=self.alpha)

        super().__init__(loss_fn=custom_loss_function)
        custom_loss_function.params = [x for x in self.policy_net.parameters() if x.dim() == 2]

    def file_display_name(self):
        if not self.embed_alpha_in_filename:
            return super().file_display_name()

        return f"{super().file_display_name()}_{self.alpha}"


class ReluActivation(BaseAgent):

    display_name = "ReluActivation"

    def __init__(self):
        network = nn.Sequential(
            nn.Flatten(1, -1), nn.Linear(100, 10), nn.ReLU(), nn.Linear(10, 3)
        )

        super().__init__(network=network)


class ReluLossActivation(BaseAgent):

    display_name = "Positive Loss Agent + ReluActivation"

    def __init__(self):
        from blue_ai.envs.custom_decay import PositivePenaltyLoss

        custom_loss_function = PositivePenaltyLoss(alpha=0.2)
        network = nn.Sequential(
            nn.Flatten(1, -1), nn.Linear(100, 10), nn.ReLU(), nn.Linear(10, 3)
        )

        super().__init__(loss_fn=custom_loss_function, network=network)

        custom_loss_function.policy_hook = self.policy_net


class WeightDropAgent(BaseAgent):
    display_name = "Weight Dropout as spineloss "

    def __init__(self, p=0.0):
        network = nn.Sequential(
            nn.Flatten(1, -1),
            SpinelossLayer(100, 10, dropout_rate=p),
            nn.ReLU(),
            nn.Linear(10, 3)
        )

        super().__init__(network=network)