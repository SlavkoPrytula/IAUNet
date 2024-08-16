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
