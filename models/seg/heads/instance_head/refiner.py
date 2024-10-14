import torch 
from torch import nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

import sys
sys.path.append('.')

from .iam import (IAM)

from configs import cfg
from utils.registry import HEADS

    

@HEADS.register(name="Refiner")
class Refiner(nn.Module):
    def __init__(self, 
                 in_channels: int = 256, 
                 num_convs: int = 4, 
                 num_classes: int = 80, 
                 kernel_dim: int = 256, 
                 num_masks: int = 100, 
                 num_groups: int = 1,
                 activation: str = "softmax"):
        super().__init__()
        self.dim = in_channels
        self.num_convs = num_convs
        self.num_masks = num_masks
        self.kernel_dim = kernel_dim
        self.num_groups = num_groups
        self.num_classes = num_classes + 1
        self.activation = activation
        self.scale_factor = 1
        
        # iam prediction, a simple conv
        self.iam_conv = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        self.refiner = nn.Conv2d(self.num_masks * 2, self.num_masks, kernel_size=1) 
        
        expand_dim = self.dim * self.num_groups
        self.fc = nn.Linear(expand_dim, expand_dim)
        self.fc_fuse = nn.Linear(expand_dim*2, expand_dim)

        # Outputs
        self.cls_score = nn.Linear(expand_dim, self.num_classes)
        self.mask_kernel = nn.Linear(expand_dim, self.kernel_dim)
        self.objectness = nn.Linear(expand_dim, 1)
        self.bbox_pred = nn.Linear(expand_dim, 4)
        
        self.prior_prob = 0.01
        self._init_weights()
        
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))
        print("refiner-v1.1")


    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, bias_value)

        init.normal_(self.refiner.weight, std=0.01)
        init.constant_(self.refiner.bias, bias_value)

        c2_xavier_fill(self.fc)
        c2_xavier_fill(self.fc_fuse)


    def forward(self, features, prev_iam=None, last_stage=False):
        iam = self.iam_conv(features)
        B, N, H, W = iam.shape
        C = features.size(1)

        if prev_iam is not None:
            prev_iam = nn.UpsamplingBilinear2d(scale_factor=2)(prev_iam)
            iam = torch.cat([iam, prev_iam], 1)
            iam = self.refiner(iam)
        
        if last_stage:
            if self.activation == "softmax":
                iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)
            else:
                raise NotImplementedError(f"No activation {self.activation} found!")
            
            inst_features = torch.bmm(iam_prob, features.view(B, C, -1).permute(0, 2, 1))
            inst_features = inst_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
            
            inst_features = F.relu_(self.fc(inst_features))

            # predictions.
            pred_logits = self.cls_score(inst_features)
            pred_mask_kernel = self.mask_kernel(inst_features)
            pred_scores = self.objectness(inst_features)
            pred_bboxes = self.bbox_pred(inst_features)

            results = {
                'logits': pred_logits,
                'mask_kernel': pred_mask_kernel,
                'objectness_scores': pred_scores,
                'bboxes': pred_bboxes,
                'iam': iam,
                'inst_feats': inst_features
            }

            return results
    
        results = {
            'iam': iam,
        }
        return results
