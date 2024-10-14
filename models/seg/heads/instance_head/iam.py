import torch 
from torch import nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

import sys
sys.path.append("./")


class IAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super(IAM, self).__init__()
        self.iam_conv = nn.Conv2d(in_channels, out_channels, 
                                  kernel_size=kernel_size, stride=stride, padding=padding, 
                                  groups=groups)
        
        self.prior_prob = 0.01
        self.init_weights()

    def init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        init.normal_(self.iam_conv.weight, std=0.01)
        init.constant_(self.iam_conv.bias, bias_value)

    def forward(self, x):
        x = self.iam_conv(x)
        return x



class DeepIAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super(DeepIAM, self).__init__()
        self.iam_conv1 = nn.Conv2d(in_channels, in_channels, 
                                   kernel_size=kernel_size, stride=stride, padding=padding, 
                                   groups=groups)
        self.gelu = nn.GELU()
        self.iam_conv2 = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=kernel_size, stride=stride, padding=padding, 
                                   groups=groups)
        
        self.prior_prob = 0.01
        self.init_weights()

    def init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        init.normal_(self.iam_conv1.weight, std=0.01)
        init.constant_(self.iam_conv1.bias, bias_value)
        init.normal_(self.iam_conv2.weight, std=0.01)
        init.constant_(self.iam_conv2.bias, bias_value)

    def forward(self, x):
        x = self.iam_conv1(x)
        x = self.gelu(x)
        x = self.iam_conv2(x)
        return x