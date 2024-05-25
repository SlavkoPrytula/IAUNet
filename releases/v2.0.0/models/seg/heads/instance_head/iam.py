import torch 
from torch import nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

import math
from configs import cfg



class IAM(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(IAM, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)
        
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


class DeepIAM(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(DeepIAM, self).__init__()
        self.num_convs = 2
        
        convs = []
        for _ in range(self.num_convs):
            convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups))
            convs.append(nn.BatchNorm2d(out_channels))
            convs.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        self.convs = nn.Sequential(*convs)
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)

        self._init_weights()

    def _init_weights(self):
        for m in self.convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        c2_msra_fill(self.projection)

    def forward(self, x):
        x = self.convs(x)
        x = self.projection(x)
        
        return x
