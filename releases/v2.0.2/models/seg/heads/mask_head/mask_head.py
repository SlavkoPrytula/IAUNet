import torch 
from torch import nn
from torch.nn import init
import numpy as np

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

import sys
sys.path.append('.')

# from ..common import _make_stack_3x3_convs
from models.seg.heads.common import _make_stack_3x3_convs


class MaskBranch(nn.Module):
    def __init__(self, in_channels, out_channels=256, kernel_dim=128, num_convs=4):
        super().__init__()
        dim = out_channels
        num_convs = num_convs
        kernel_dim = kernel_dim
        
        self.mask_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        # self.projection = nn.Conv2d(dim, kernel_dim, kernel_size=1)

        # self.projection = nn.Sequential(
        #     nn.Conv2d(
        #         dim, kernel_dim,
        #         kernel_size=1, stride=1,
        #         padding=0),
        #     nn.BatchNorm2d(kernel_dim),
        #     nn.ReLU(inplace=True)
        # )

        self._init_weights()

    def _init_weights(self):
        for m in self.mask_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

        # for m in self.projection.modules():
        #     if isinstance(m, nn.Conv2d):
        #         c2_msra_fill(m)
        # c2_msra_fill(self.projection)

    def forward(self, features):
        # mask features (x4 convs)
        features = self.mask_convs(features)
        # features = self.projection(features)
        return features

    
    
if __name__ == '__main__':
    from configs import cfg
    
    mask_decoder = MaskBranch(32).to(cfg.device)
    x = torch.randn(2, 32, 64, 64).to(cfg.device)

    out = mask_decoder(x)
    print(out.shape)
