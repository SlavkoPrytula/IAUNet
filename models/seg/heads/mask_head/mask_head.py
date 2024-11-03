import torch 
from torch import nn
from fvcore.nn.weight_init import c2_msra_fill

import sys
sys.path.append('.')

from ..common import _make_stack_3x3_convs
from models.seg.nn.blocks import (DoubleConv_v1)
from utils.registry import HEADS


@HEADS.register(name='MaskDoubleConv')
class MaskBranch(nn.Module):
    def __init__(self, in_channels, out_channels=256, num_convs=4):
        super().__init__()
        layers = [DoubleConv_v1(in_channels if i == 0 else out_channels, out_channels) 
                  for i in range(num_convs)]
        self.mask_convs = nn.Sequential(
            *layers,
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mask_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

    def forward(self, features):
        features = self.mask_convs(features)
        return features
    

@HEADS.register(name='MaskStackedConv')
class MaskBranch(nn.Module):
    def __init__(self, in_channels, out_channels=256, num_convs=4):
        super().__init__()
        self.mask_convs = _make_stack_3x3_convs(num_convs, in_channels, out_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.mask_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

    def forward(self, features):
        features = self.mask_convs(features)
        return features



@HEADS.register(name='LightMaskStackedConv')
class MaskBranch(nn.Module):
    def __init__(self, in_channels, out_channels=256, num_convs=4):
        super(MaskBranch, self).__init__()
        self.c_in = in_channels
        self.c_out = out_channels
        hidden = out_channels
        
        self.depthwise_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise_conv1 = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.norm_relu1 = nn.Sequential(
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        
        self.depthwise_conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.pointwise_conv2 = nn.Conv2d(hidden, out_channels, kernel_size=1)
        self.norm_relu2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        out = self.depthwise_conv1(x)
        out = self.pointwise_conv1(out)
        out = self.norm_relu1(out)

        out = self.depthwise_conv2(out)
        out = self.pointwise_conv2(out)
        out = self.norm_relu2(out)
        
        out = out + self.projection(x)
        
        return out
    
