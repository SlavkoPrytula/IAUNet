import torch 
from torch import nn
from torch.nn import init
import numpy as np

from fvcore.nn.weight_init import c2_msra_fill

import sys
sys.path.append('.')

from ..common import _make_stack_3x3_convs
from models.seg.nn.blocks import (DoubleConv_v1, DoubleConv_v3_1)


class MaskBranch(nn.Module):
    def __init__(self, in_channels, out_channels=256, num_convs=4):
        super().__init__()
        
        # self.mask_convs = _make_stack_3x3_convs(num_convs, in_channels, out_channels)
        self.mask_convs = nn.Sequential(
            DoubleConv_v1(in_channels, out_channels), 
            DoubleConv_v1(out_channels, out_channels), 
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

    
    
if __name__ == '__main__':
    from configs import cfg
    
    mask_decoder = MaskBranch(32).to(cfg.device)
    x = torch.randn(2, 32, 64, 64).to(cfg.device)

    out = mask_decoder(x)
    print(out.shape)
