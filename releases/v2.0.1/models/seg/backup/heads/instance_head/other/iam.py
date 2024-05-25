import torch 
from torch import nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F

import math
from configs import cfg



class IAM(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super(IAM, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.prior_prob = 0.01
        self._init_weights()

    def _init_weights(self):
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.conv]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.conv.weight, std=0.01)

    def forward(self, x):
        x = self.conv(x)
        
        return x
