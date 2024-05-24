from copy import Error
import torch.nn as nn
import torch
from torch.nn.modules import MSELoss, Sequential

from tqdm.gui import tqdm


class ExponentialLoss(nn.Module):
    known = MSELoss()

    def __init__(
        self,
    ):
        super(ExponentialLoss, self).__init__()

        self.policy_hook: Sequential = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        net = next(self.policy_hook.parameters())
        positive_penalty = 0.2 * torch.clamp(net, min=0).sum()
        normal = self.known(inputs, targets)

        return normal + positive_penalty
