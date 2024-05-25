import torch 
from torch import digamma, nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F

# import math
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

import sys
sys.path.append('.')

# from models.seg.heads.common import _make_stack_3x3_convs, MLP
from .iam import (IAM_ASPP, IAM)
from ..common import _make_stack_3x3_convs, MLP
from ...nn.blocks import DoubleConv_v2
from configs import cfg
from utils.registry import HEADS
    
    

class PriorInstanceBranch(nn.Module):
    def __init__(self, in_channels, out_channels=256, num_convs=4):
        super().__init__()
        
        self.inst_convs = _make_stack_3x3_convs(num_convs, in_channels, out_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

    def forward(self, features):
        features = self.inst_convs(features)
        return features



# InstanceHead_v1.1
@HEADS.register(name="InstanceBranch_v1.1")
class InstanceBranch(nn.Module):
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
        self.iam_conv = IAM(self.dim, self.num_masks * self.num_groups, self.num_groups)
        
        expand_dim = self.dim * self.num_groups
        self.fc = nn.Linear(expand_dim, expand_dim)
        self.fc_fuse = nn.Linear(expand_dim*2, expand_dim) #MLP(expand_dim*2, expand_dim*2, expand_dim, 3)

        # Outputs
        self.cls_score = nn.Linear(expand_dim, self.num_classes)
        self.mask_kernel = nn.Linear(expand_dim, self.kernel_dim)
        self.border_kernel = nn.Linear(expand_dim, self.kernel_dim)
        self.objectness = nn.Linear(expand_dim, 1)
        self.bbox_pred = nn.Linear(expand_dim, 4)
        
        self.prior_prob = 0.01
        self._init_weights()
        
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))
        print("v1.1")


    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, bias_value)

        init.normal_(self.border_kernel.weight, std=0.01)
        init.constant_(self.border_kernel.bias, 0.0)

        c2_xavier_fill(self.fc)
        c2_xavier_fill(self.fc_fuse)


    def forward(self, features, prev_inst_features=None):
        iam = self.iam_conv(features)
        B, N, H, W = iam.shape
        C = features.size(1)
        
        if self.activation == "softmax":
            iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)
        else:
            raise NotImplementedError(f"No activation {self.activation} found!")
        
        inst_features = torch.bmm(iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        inst_features = inst_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
        if prev_inst_features is not None:
            inst_features = torch.cat([inst_features, prev_inst_features], -1)
            inst_features = F.relu_(self.fc_fuse(inst_features))
            # inst_features = self.fc_fuse(inst_features)
        
        inst_features = F.relu_(self.fc(inst_features))

        # predictions.
        pred_logits = self.cls_score(inst_features)
        pred_mask_kernel = self.mask_kernel(inst_features)
        pred_border_kernel = self.border_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        pred_bboxes = self.bbox_pred(inst_features)

        results = {
            'logits': pred_logits,
            'mask_kernel': pred_mask_kernel,
            'border_kernel': pred_border_kernel,
            'objectness_scores': pred_scores,
            'bboxes': pred_bboxes,
            'iam': iam,
            'inst_feats': inst_features
        }

        return results
    


@HEADS.register(name="InstanceBranch-v1.2-occluders")
class InstanceBranch(nn.Module):
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
        self.inst_iam_conv = IAM(self.dim, self.num_masks * self.num_groups, self.num_groups)
        self.occl_iam_conv = IAM(self.dim, self.num_masks * self.num_groups, self.num_groups)
        
        expand_dim = self.dim * self.num_groups
        self.fc = nn.Linear(expand_dim*2, expand_dim)
        # self.fc_fuse = nn.Linear(expand_dim*2, expand_dim) #MLP(expand_dim*2, expand_dim*2, expand_dim, 3)

        # outputs.
        self.cls_score = nn.Linear(expand_dim, self.num_classes)
        self.mask_kernel = nn.Linear(expand_dim, self.kernel_dim)
        self.occluder_kernel = nn.Linear(expand_dim, self.kernel_dim)
        self.overlap_kernel = nn.Linear(expand_dim, self.kernel_dim)
        self.objectness = nn.Linear(expand_dim, 1)
        self.bbox_pred = nn.Linear(expand_dim, 4)
        
        self.prior_prob = 0.01
        self._init_weights()
        
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))
        print("v1.2-occluders")


    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, bias_value)

        init.normal_(self.occluder_kernel.weight, std=0.01)
        init.constant_(self.occluder_kernel.bias, 0.0)

        init.normal_(self.overlap_kernel.weight, std=0.01)
        init.constant_(self.overlap_kernel.bias, 0.0)

        c2_xavier_fill(self.fc)
        # c2_xavier_fill(self.fc_fuse)


    def forward(self, features, prev_inst_features=None):
        inst_iam = self.inst_iam_conv(features)
        occl_iam = self.occl_iam_conv(features)

        B, N, H, W = inst_iam.shape
        C = features.size(1)
        
        if self.activation == "softmax":
            inst_iam_prob = F.softmax(inst_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
            occl_iam_prob = F.softmax(occl_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
        else:
            raise NotImplementedError(f"No activation {self.activation} found!")
        
        inst_features = torch.bmm(inst_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        inst_features = inst_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        occl_features = torch.bmm(occl_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        occl_features = occl_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        
        # if prev_inst_features is not None:
        #     inst_features = torch.cat([inst_features, prev_inst_features], -1)
        #     inst_features = F.relu_(self.fc_fuse(inst_features))
        
        inst_features = torch.cat([inst_features, occl_features], -1)
        inst_features = F.relu_(self.fc(inst_features))

        # predictions.
        pred_logits = self.cls_score(inst_features)
        pred_mask_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        pred_bboxes = self.bbox_pred(inst_features)

        pred_occluder_kernel = self.occluder_kernel(occl_features)
        pred_overlap_kernel = self.overlap_kernel(occl_features)

        results = {
            'logits': pred_logits,
            'mask_kernel': pred_mask_kernel,
            "occluder_kernel": pred_occluder_kernel,
            "overlap_kernel": pred_overlap_kernel,
            'objectness_scores': pred_scores,
            'bboxes': pred_bboxes,
            'iam': inst_iam,
            'inst_feats': inst_features
        }

        return results



# InstanceHead_v3-multiheaded
# introduces multiple activations per object for better features
@HEADS.register(name="InstanceBranch-v3-multiheaded")
class InstanceBranch(nn.Module):
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
        
        self.iam_conv = nn.Conv2d(self.dim, self.num_masks * self.num_groups, 3, padding=1, stride=1)
        
        expand_dim = self.dim * self.num_groups
        self.fc = nn.Linear(expand_dim, expand_dim)
        self.fc_fuse = nn.Linear(expand_dim*2, expand_dim)

        self.cls_score = nn.Linear(expand_dim, self.num_classes)
        self.mask_kernel = nn.Linear(expand_dim, self.kernel_dim)
        self.objectness = nn.Linear(expand_dim, 1)
        self.bbox_pred = nn.Linear(expand_dim, 4)
        
        self.prior_prob = 0.01
        self._init_weights()
        
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))
        print("v3-multiheaded")


    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.iam_conv, self.cls_score]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.iam_conv.weight, std=0.01)
        init.normal_(self.cls_score.weight, std=0.01)
        
        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)

        init.normal_(self.bbox_pred.weight, std=0.01)
        init.constant_(self.bbox_pred.bias, 0.0)
        
        c2_xavier_fill(self.fc)
        c2_xavier_fill(self.fc_fuse)


    def forward(self, features, prev_inst_features=None):
        B, C, H, W = features.size()
        
        iam = self.iam_conv(features)
        iam = iam.view(B, self.num_groups, self.num_masks, H, W)
        
        inst_features = []
        iams = []
        for g in range(self.num_groups):
            group_iam = iam[:, g, :, :, :]
            
            if self.activation == "softmax":
                iam_prob = F.softmax(group_iam.view(B, self.num_masks, -1) + self.softmax_bias, dim=-1)
            else:
                raise NotImplementedError(f"No activation {self.activation} found!")

            # Aggregate features: BxCxHxW -> Bx(HW)xC
            inst_feats = torch.bmm(iam_prob, features.view(B, C, -1).permute(0, 2, 1))
            inst_features.append(inst_feats)
            iams.append(group_iam)
        
        inst_features = torch.cat(inst_features, -1)
        iams = torch.cat(iams, dim=1)

        if prev_inst_features is not None:
            inst_features = torch.cat([inst_features, prev_inst_features], -1)
            inst_features = F.relu_(self.fc_fuse(inst_features))
            # inst_features = self.fc_fuse(inst_features)

        inst_features = F.relu_(self.fc(inst_features))

        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        pred_bboxes = self.bbox_pred(inst_features)


        results = {
            'logits': pred_logits,
            'mask_kernel': pred_kernel,
            'border_kernel': pred_kernel,
            'objectness_scores': pred_scores,
            'bboxes': pred_bboxes,
            'iam': iams,
            'inst_feats': inst_features
        }
        
        return results




if __name__ == "__main__":
    instance_head = InstanceBranch(in_channels=256, num_classes=1, kernel_dim=10, num_masks=5, num_groups=2)
    x = torch.rand(1, 256, 512, 512)
    pred_logits, pred_kernel, pred_scores, iam = instance_head(x)
    print(pred_kernel.shape)
    print(pred_logits.shape)