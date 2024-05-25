import torch 
from torch import digamma, nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill
from torchvision.ops.deform_conv import DeformConv2d

import sys
sys.path.append('.')


from .instance_head import InstanceBranch
from configs import cfg
from utils.registry import HEADS
    

@HEADS.register(name="DeformInstanceBranch")
class DeformInstanceBranch(InstanceBranch):
    def __init__(self, 
                 in_channels: int = 256, 
                 num_convs: int = 4, 
                 num_classes: int = 80, 
                 kernel_dim: int = 256, 
                 num_masks: int = 100, 
                 num_groups: int = 1,
                 activation: str = "softmax"):
        super(DeformInstanceBranch, self).__init__(
            in_channels=in_channels, 
            num_convs=num_convs, 
            num_classes=num_classes, 
            kernel_dim=kernel_dim, 
            num_masks=num_masks, 
            num_groups=num_groups,
            activation=activation
        )

        self.iam_conv = DeformConv2d(self.dim, self.num_masks * self.num_groups, 3, 
                                     padding=1, groups=self.num_groups)
        

if __name__ == "__main__":
    instance_head = DeformInstanceBranch(in_channels=256, num_classes=1, kernel_dim=10, num_masks=5)
    x = torch.rand(1, 256, 512, 512)
    pred_logits, pred_kernel, pred_scores, iam = instance_head(x)
    print(pred_kernel.shape)
    print(pred_logits.shape)
