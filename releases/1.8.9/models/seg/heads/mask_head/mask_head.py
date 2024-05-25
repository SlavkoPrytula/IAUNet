import torch 
from torch import nn
from torch.nn import init
import numpy as np

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from models.seg.heads.common import _make_stack_3x3_convs


class MaskBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        dim = 256
        num_convs = 4
        kernel_dim = 128
        
        self.mask_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        self.projection = nn.Conv2d(dim, kernel_dim, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.mask_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        c2_msra_fill(self.projection)

    def forward(self, features):
        # mask features (x4 convs)
        features = self.mask_convs(features)
        return self.projection(features)
    
    
if __name__ == '__main__':
    from configs import cfg
    
    mask_decoder = MaskBranch(32).to(cfg.device)
    x = torch.randn(2, 32, 64, 64).to(cfg.device)

    out = mask_decoder(x)
    print(out.shape)
