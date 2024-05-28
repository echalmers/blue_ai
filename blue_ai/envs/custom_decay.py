import torch.nn as nn
import torch
from torch.nn.modules import MSELoss, Sequential


class PositivePenaltyLoss(nn.Module):
    known = MSELoss()

    """
    Uses a combinations of MSE Loss with a RELU penalty, meaning that positive 
    weights will always have a large penalty 
    """

    def __init__(self, alpha=0.2):
        super(PositivePenaltyLoss, self).__init__()
        self.policy_hook: Sequential = None
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        net = next(self.policy_hook.parameters())
        # This is actually a ReLU function
        # https://discuss.pytorch.org/t/why-are-there-3-relu-functions-or-maybe-even-more/5891/2
        positive_penalty = torch.clamp(net, min=0).sum()
        normal = self.known(inputs, targets)

        return normal + self.alpha * positive_penalty
