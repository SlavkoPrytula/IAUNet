import torch 
from torch import nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F

import math
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

import sys
sys.path.append('.')

from models.seg.heads.common import _make_stack_3x3_convs
from configs import cfg
    
    

class PriorInstanceBranch(nn.Module):
    def __init__(self, in_channels, out_channels=256, num_convs=4):
        super().__init__()
        dim = out_channels
        num_convs = num_convs
        
        self.inst_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

    def forward(self, features):
        # instance features (x4 convs)
        features = self.inst_convs(features)
        return features
    


# NOTE: [main]
class InstanceBranch(nn.Module):
    def __init__(self, dim, kernel_dim, num_masks=10):
        super().__init__()
        dim = dim
        num_masks = num_masks
        kernel_dim = kernel_dim
        
        self.num_groups = 1
        self.num_classes = 1
        
        # iam prediction, a simple conv
        self.iam_conv = nn.Sequential(
            nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)
        )
        self.inst_conv = nn.Conv2d(dim, dim, 3, padding=1)
        
        expand_dim = dim * self.num_groups
        self.fc = nn.Linear(expand_dim, expand_dim)

        # outputs
        self.cls_score = nn.Linear(expand_dim, self.num_classes)
        self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
        self.objectness = nn.Linear(expand_dim, 1)
        
        self.prior_prob = 0.01
        self._init_weights()
        
        self.softmax_bias = nn.Parameter(torch.ones([1, ]), requires_grad=True)
        self.temprature = 30


    def _init_weights(self):
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.cls_score]:
            init.constant_(module.bias, bias_value)

        for modules in [self.iam_conv, self.inst_conv]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, bias_value)

        init.normal_(self.cls_score.weight, std=0.01)
        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)
        c2_xavier_fill(self.fc)


    def forward(self, features, idx=None):
        # predict instance activation maps
        iam = self.iam_conv(features)

        B, N, H, W = iam.shape
        C = features.size(1)
        
        # BxNxHxW -> BxNx(HW)
        # iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)
        # iam_prob = F.softmax(iam.view(B, N, -1) / self.temprature, dim=-1)
        iam_prob = F.softmax(iam.view(B, N, -1), dim=-1)


        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(
            iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        
        # nxhw * hwxc -> n*c
        inst_features = inst_features.reshape(
            B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
        inst_features = F.relu_(self.fc(inst_features))

        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)

        return pred_logits, pred_kernel, pred_scores, iam


