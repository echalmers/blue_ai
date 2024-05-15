from typing import List, Any
import torch.nn as nn
import torch
from torch.nn.modules import MSELoss


class ExponentialLoss(nn.Module):

    def __init__(self, lambda_reg=0):
        super(ExponentialLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        return torch.exp(targets - inputs - 2).median()
