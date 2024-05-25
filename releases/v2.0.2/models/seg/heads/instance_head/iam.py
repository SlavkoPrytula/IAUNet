import torch 
from torch import nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

import sys
sys.path.append("./")

import math
from configs import cfg

from models.seg.blocks import DynamicConv



class IAM(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(IAM, self).__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        )
        
        self.prior_prob = 0.01
        self._init_weights()

    def _init_weights(self):
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)

        for modules in [self.conv_in, self.conv_out]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.normal_(l.weight, std=0.1)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, bias_value)

        # for module in self.conv:
        #     init.constant_(module.bias, bias_value)
        # init.normal_(self.conv.weight, std=0.01)

    def forward(self, x):
        x = self.conv_in(x)
        out = x
        x = self.conv_out(x)
        x = x + out

        return x




# class IAM(nn.Module):
#     def __init__(self, in_channels, out_channels, groups=1):
#         super(IAM, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)
        
#         self.prior_prob = 0.01
#         self._init_weights()

#     def _init_weights(self):
#         bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)

#         for modules in [self.conv]:
#             for l in modules.modules():
#                 if isinstance(l, nn.Conv2d):
#                     nn.init.normal_(l.weight, std=0.1)
#                     if l.bias is not None:
#                         nn.init.constant_(l.bias, bias_value)

#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class IAM(nn.Module):
#     def __init__(self, in_channels, out_channels, groups=1):
#         super(IAM, self).__init__()
#         self.conv = DynamicConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        
#         self.prior_prob = 0.01
#         self._init_weights()

#     def _init_weights(self):
#         bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)

#         for modules in [self.conv]:
#             for l in modules.modules():
#                 if isinstance(l, nn.Conv2d):
#                     nn.init.normal_(l.weight, std=0.1)
#                     if l.bias is not None:
#                         nn.init.constant_(l.bias, bias_value)

#     def forward(self, x):
#         x = self.conv(x)
        
#         return x


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


if __name__ == '__main__':
    x = torch.randn(2, 32, 64, 64)
    iam = IAM(32, 10)
    out = iam(x)
    print(out.shape)