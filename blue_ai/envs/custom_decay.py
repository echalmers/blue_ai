import torch.nn as nn
import torch
from torch.nn.modules import MSELoss, Sequential


class PositivePenaltyLoss(nn.Module):
    known = MSELoss()
    leaky_relu = torch.nn.LeakyReLU(negative_slope=0)

    """
    Uses a combinations of MSE Loss with a RELU penalty, meaning that positive 
    weights will always have a large penalty 
    """

    def __init__(self, alpha=0.2):

        super(PositivePenaltyLoss, self).__init__()
        self.params = None
        self.λ = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        positive_penalty = 0
        for paramset in self.params:
            if paramset.dim() == 2:
                positive_penalty += (self.leaky_relu(paramset) ** 2).sum()
        normal = self.known(inputs, targets)

        return normal + self.λ * positive_penalty
