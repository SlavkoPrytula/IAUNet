import torch 
from torch import nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

import sys
sys.path.append('.')

from .iam import (IAM, AttentionIAM)
from ..common import _make_stack_3x3_convs
from models.seg.nn.blocks import (DoubleConv_v1, DoubleConv_v3_1)
from models.seg.nn.blocks import MLP

from configs import cfg
from utils.registry import HEADS

    
    

class InstanceBranch(nn.Module):
    def __init__(self, in_channels, out_channels=256, num_convs=4):
        super().__init__()
        
        # self.inst_convs = _make_stack_3x3_convs(num_convs, in_channels, out_channels)
        self.inst_convs = nn.Sequential(
            DoubleConv_v1(in_channels, out_channels), 
            DoubleConv_v1(out_channels, out_channels), 
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.inst_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

    def forward(self, features):
        features = self.inst_convs(features)
        return features



# InstanceHead-v1.1
# base version, single activation map per object
@HEADS.register(name="InstanceHead-v1.1")
class InstanceHead(nn.Module):
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
        
        expand_dim = self.dim * self.num_groups
        self.fc = nn.Linear(expand_dim, expand_dim)
        # self.fc_fuse = nn.Linear(expand_dim*2, expand_dim)
        self.fc_fuse = MLP(expand_dim*2, expand_dim*4, expand_dim, 2)

        # Outputs
        self.cls_score = nn.Linear(expand_dim, self.num_classes)
        self.mask_kernel = nn.Linear(expand_dim, self.kernel_dim)
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

        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)

        c2_xavier_fill(self.fc)
        # c2_xavier_fill(self.fc_fuse)


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



# TODO: add mask_kernel init to all.
@HEADS.register(name="InstanceHead-v1.1-multi-iam")
class InstanceHead(nn.Module):
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
        
        # IAM prediction, a simple conv
        self.iam_conv = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        
        # Additional IAM prediction with larger kernel
        self.iam_conv_large = IAM(self.dim, self.num_masks * self.num_groups, kernel_size=7, padding=3, groups=self.num_groups)
        
        expand_dim = self.dim * self.num_groups
        self.fc = nn.Linear(expand_dim * 2, expand_dim * 2)
        self.fc_fuse = nn.Linear(expand_dim * 4, expand_dim * 2) #MLP(expand_dim*4, expand_dim*4, expand_dim*2, 3)

        # Outputs
        self.cls_score = nn.Linear(expand_dim * 2, self.num_classes)
        self.mask_kernel = nn.Linear(expand_dim * 2, self.kernel_dim)
        self.border_kernel = nn.Linear(expand_dim * 2, self.kernel_dim)
        self.objectness = nn.Linear(expand_dim * 2, 1)
        self.bbox_pred = nn.Linear(expand_dim * 2, 4)
        
        self.prior_prob = 0.01
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))
        self._init_weights()
        print("v1.1-multi-iam")


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
        iam_large = self.iam_conv_large(features)
        B, N, H, W = iam.shape
        C = features.size(1)

        iams = torch.cat([iam, iam_large], dim=1)
        
        if self.activation == "softmax":
            iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)
            iam_large_prob = F.softmax(iam_large.view(B, N, -1), dim=-1)
        elif self.activation == "sigmoid":
            iam_prob = iam.sigmoid()
            iam_prob = iam_prob.view(B, N, -1)
            normalizer = iam_prob.sum(-1).clamp(min=1e-6)
            iam_prob = iam_prob / normalizer[:, :, None]
            
            iam_large_prob = iam_large.sigmoid()
            iam_large_prob = iam_large_prob.view(B, N, -1)
            normalizer_large = iam_large_prob.sum(-1).clamp(min=1e-6)
            iam_large_prob = iam_large_prob / normalizer_large[:, :, None]
        else:
            raise NotImplementedError(f"No activation {self.activation} found!")
        
        inst_features = torch.bmm(iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        inst_features = inst_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
        inst_features_large = torch.bmm(iam_large_prob, features.view(B, C, -1).permute(0, 2, 1))
        inst_features_large = inst_features_large.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        # Concatenate instance features from both IAMs
        inst_features = torch.cat([inst_features, inst_features_large], dim=-1)
        
        if prev_inst_features is not None:
            inst_features = torch.cat([inst_features, prev_inst_features], -1)
            inst_features = F.relu_(self.fc_fuse(inst_features))
        
        inst_features = F.relu_(self.fc(inst_features))

        # Predictions
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
            'iam': iams,
            'inst_feats': inst_features
        }

        return results



@HEADS.register(name="InstanceHead-v1.2-occluders")
class InstanceHead(nn.Module):
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
        
        # IAM prediction for instances and occluders
        self.inst_iam_conv = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        self.occl_iam_conv = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        
        expand_dim = self.dim * self.num_groups
        self.fc_bi_occl = nn.Linear(expand_dim, expand_dim)
        self.fc_bi_inst = nn.Linear(expand_dim, expand_dim)
        self.fc_inst_final = nn.Linear(expand_dim, expand_dim)
        self.fc_occl_final = nn.Linear(expand_dim, expand_dim)
        
        # Output layers
        self.cls_score = nn.Linear(expand_dim, self.num_classes)
        self.mask_kernel = nn.Linear(expand_dim, self.kernel_dim)
        self.occluder_kernel = nn.Linear(expand_dim, self.kernel_dim)
        self.objectness = nn.Linear(expand_dim, 1)
        self.bbox_pred = nn.Linear(expand_dim, 4)
        
        self.prior_prob = 0.01
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))
        self._init_weights()
        print("v1.2-occluders")

    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, bias_value)

        nn.init.normal_(self.occluder_kernel.weight, std=0.01)
        nn.init.constant_(self.occluder_kernel.bias, 0.0)

        nn.init.kaiming_normal_(self.fc_bi_occl.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_bi_inst.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_inst_final.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_occl_final.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, features, prev_inst_features=None):
        inst_iam = self.inst_iam_conv(features)
        occl_iam = self.occl_iam_conv(features)

        B, N, H, W = inst_iam.shape
        C = features.size(1)
        
        if self.activation == "softmax":
            inst_iam_prob = F.softmax(inst_iam.view(B, N, -1), dim=-1)
            occl_iam_prob = F.softmax(occl_iam.view(B, N, -1), dim=-1)
        else:
            raise NotImplementedError(f"No activation {self.activation} found!")
        
        inst_features = torch.bmm(inst_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        inst_features = inst_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        occl_features = torch.bmm(occl_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        occl_features = occl_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        inst_features = inst_features + F.relu_(self.fc_bi_inst(occl_features))
        occl_features = occl_features + F.relu_(self.fc_bi_occl(inst_features))

        inst_features = F.relu_(self.fc_inst_final(inst_features))
        occl_features = F.relu_(self.fc_occl_final(occl_features))

        # Output predictions.
        pred_logits = self.cls_score(inst_features)
        pred_mask_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        pred_bboxes = self.bbox_pred(inst_features)

        pred_occluder_kernel = self.occluder_kernel(occl_features)

        results = {
            'logits': pred_logits,
            'mask_kernel': pred_mask_kernel,
            "occluder_kernel": pred_occluder_kernel,
            # "overlap_kernel": pred_occluder_kernel,
            'objectness_scores': pred_scores,
            'bboxes': pred_bboxes,
            'iam': inst_iam,
            'inst_feats': inst_features
        }

        return results



@HEADS.register(name="InstanceHead-v1.3-overlaps")
class InstanceHead(nn.Module):
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
        
        # IAM prediction for instances and overlaps
        self.inst_iam_conv = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        self.overlap_iam_conv = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        
        expand_dim = self.dim * self.num_groups
        self.fc_inst = nn.Linear(expand_dim*2, expand_dim)
        self.fc_overlap = nn.Linear(expand_dim, expand_dim)
        
        # Output layers
        self.cls_score = nn.Linear(expand_dim, self.num_classes)
        self.mask_kernel = nn.Linear(expand_dim, self.kernel_dim)
        self.overlap_kernel = nn.Linear(expand_dim, self.kernel_dim)
        self.objectness = nn.Linear(expand_dim, 1)
        self.bbox_pred = nn.Linear(expand_dim, 4)
        
        self.prior_prob = 0.01
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))
        self._init_weights()
        print("v1.3-overlaps")

    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, bias_value)

        nn.init.normal_(self.overlap_kernel.weight, std=0.01)
        nn.init.constant_(self.overlap_kernel.bias, 0.0)

        c2_xavier_fill(self.fc_inst)
        c2_xavier_fill(self.fc_overlap)

    def forward(self, features, prev_inst_features=None):
        inst_iam = self.inst_iam_conv(features)
        overlap_iam = self.overlap_iam_conv(features)

        B, N, H, W = inst_iam.shape
        C = features.size(1)
        
        if self.activation == "softmax":
            inst_iam_prob = F.softmax(inst_iam.view(B, N, -1), dim=-1)
            overlap_iam_prob = F.softmax(overlap_iam.view(B, N, -1), dim=-1)
        else:
            raise NotImplementedError(f"No activation {self.activation} found!")
        
        inst_features = torch.bmm(inst_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        inst_features = inst_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        overlap_features = torch.bmm(overlap_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        overlap_features = overlap_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        inst_features = torch.cat([inst_features, overlap_features], -1)
        iams = torch.cat([inst_iam, overlap_iam], dim=1)

        inst_features = F.relu_(self.fc_inst(inst_features))
        overlap_features = F.relu_(self.fc_overlap(overlap_features))

        # output predictions.
        pred_logits = self.cls_score(inst_features)
        pred_mask_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        pred_bboxes = self.bbox_pred(inst_features)
        pred_overlap_kernel = self.overlap_kernel(overlap_features)

        results = {
            'logits': pred_logits,
            'mask_kernel': pred_mask_kernel,
            "overlap_kernel": pred_overlap_kernel,
            'objectness_scores': pred_scores,
            'bboxes': pred_bboxes,
            'iam': iams,
            'inst_feats': inst_features
        }

        return results



# InstanceHead_v3-multiheaded
# introduces multiple activations per object for better features
@HEADS.register(name="InstanceHead-v3-multiheaded")
class InstanceHead(nn.Module):
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
        
        self.proj = nn.ModuleList([
            nn.Conv2d(self.dim, self.dim, kernel_size=1) 
            for _ in range(self.num_groups)
        ])
        
        self.iam_convs = nn.ModuleList([
            nn.Conv2d(self.dim, self.num_masks, 3, padding=1, stride=1)
            for _ in range(self.num_groups)
        ])
        
        expand_dim = self.dim * self.num_groups
        self.fc = nn.Linear(expand_dim, expand_dim)
        self.fc_fuse = nn.Linear(expand_dim * 2, expand_dim)

        self.cls_score = nn.Linear(expand_dim, self.num_classes)
        self.mask_kernel = nn.Linear(expand_dim, self.kernel_dim)
        self.objectness = nn.Linear(expand_dim, 1)
        self.bbox_pred = nn.Linear(expand_dim, 4)
        
        self.prior_prob = 0.01
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))
        self._init_weights()
        print("v3-multiheaded")

    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, bias_value)
        
        for transform, conv in zip(self.proj, self.iam_convs):
            init.normal_(transform.weight, std=0.01)
            init.constant_(transform.bias, 0.0)
            init.normal_(conv.weight, std=0.01)
            init.constant_(conv.bias, 0.0)
        
        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)

        init.normal_(self.bbox_pred.weight, std=0.01)
        init.constant_(self.bbox_pred.bias, 0.0)
        
        c2_xavier_fill(self.fc)
        c2_xavier_fill(self.fc_fuse)

    def forward(self, features, prev_inst_features=None):
        B, C, H, W = features.size()
        
        # proj.
        transformed_features = [transform(features) for transform in self.proj]

        # mha.
        conv_outputs = [conv(transformed_features[g]) for g, conv in enumerate(self.iam_convs)]
        conv_outputs = torch.stack(conv_outputs, dim=1)  # (B, G, N, H, W)
        
        inst_features = []
        iams = []
        for g in range(self.num_groups):
            group_iam = conv_outputs[:, g, :, :, :]  # (B, N, H, W)
            
            if self.activation == "softmax":
                iam_prob = F.softmax(group_iam.view(B, self.num_masks, -1), dim=-1)  # (B, N, H*W)
            else:
                raise NotImplementedError(f"No activation {self.activation} found!")

            inst_feats = torch.bmm(iam_prob, transformed_features[g].view(B, C, -1).permute(0, 2, 1))  # (B, N, C)
            inst_features.append(inst_feats)
            iams.append(group_iam)

        # cat.
        inst_features = torch.cat(inst_features, dim=-1)  # (B, N, C * G)
        iams = torch.cat(iams, dim=1)  # (B, G * N, H, W)
        
        if prev_inst_features is not None:
            inst_features = torch.cat([inst_features, prev_inst_features], dim=-1)
            inst_features = F.relu_(self.fc_fuse(inst_features))

        inst_features = F.relu_(self.fc(inst_features))

        # output predictions.
        pred_logits = self.cls_score(inst_features)
        pred_mask_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        pred_bboxes = self.bbox_pred(inst_features)

        results = {
            'logits': pred_logits,
            'mask_kernel': pred_mask_kernel,
            'objectness_scores': pred_scores,
            'bboxes': pred_bboxes,
            'iam': iams,
            'inst_feats': inst_features
        }
        
        return results




# # InstanceHead_v3-multiheaded - [depricated]
# # introduces multiple activations per object for better features
# @HEADS.register(name="InstanceHead-v3-multiheaded")
# class InstanceHead(nn.Module):
#     def __init__(self, 
#                  in_channels: int = 256, 
#                  num_convs: int = 4, 
#                  num_classes: int = 80, 
#                  kernel_dim: int = 256, 
#                  num_masks: int = 100, 
#                  num_groups: int = 1,
#                  activation: str = "softmax"):
#         super().__init__()
#         self.dim = in_channels
#         self.num_convs = num_convs
#         self.num_masks = num_masks
#         self.kernel_dim = kernel_dim
#         self.num_groups = num_groups
#         self.num_classes = num_classes + 1
#         self.activation = activation
        
#         self.iam_conv = nn.Conv2d(self.dim, self.num_masks * self.num_groups, 3, padding=1, stride=1)
        
#         expand_dim = self.dim * self.num_groups
#         self.fc = nn.Linear(expand_dim, expand_dim)
#         self.fc_fuse = nn.Linear(expand_dim*2, expand_dim)

#         self.cls_score = nn.Linear(expand_dim, self.num_classes)
#         self.mask_kernel = nn.Linear(expand_dim, self.kernel_dim)
#         self.objectness = nn.Linear(expand_dim, 1)
#         self.bbox_pred = nn.Linear(expand_dim, 4)
        
#         self.prior_prob = 0.01
#         self._init_weights()
        
#         self.softmax_bias = nn.Parameter(torch.ones([1, ]))
#         print("v3-multiheaded")


#     def _init_weights(self):
#         bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
#         for module in [self.iam_conv, self.cls_score]:
#             init.constant_(module.bias, bias_value)
#         init.normal_(self.iam_conv.weight, std=0.01)
#         init.normal_(self.cls_score.weight, std=0.01)
        
#         init.normal_(self.mask_kernel.weight, std=0.01)
#         init.constant_(self.mask_kernel.bias, 0.0)

#         init.normal_(self.bbox_pred.weight, std=0.01)
#         init.constant_(self.bbox_pred.bias, 0.0)
        
#         c2_xavier_fill(self.fc)
#         c2_xavier_fill(self.fc_fuse)


#     def forward(self, features, prev_inst_features=None):
#         B, C, H, W = features.size()
        
#         iam = self.iam_conv(features)
#         iam = iam.view(B, self.num_groups, self.num_masks, H, W)
        
#         inst_features = []
#         iams = []
#         for g in range(self.num_groups):
#             group_iam = iam[:, g, :, :, :]
            
#             if self.activation == "softmax":
#                 iam_prob = F.softmax(group_iam.view(B, self.num_masks, -1) + self.softmax_bias, dim=-1)
#             else:
#                 raise NotImplementedError(f"No activation {self.activation} found!")

#             # Aggregate features: BxCxHxW -> Bx(HW)xC
#             inst_feats = torch.bmm(iam_prob, features.view(B, C, -1).permute(0, 2, 1))
#             inst_features.append(inst_feats)
#             iams.append(group_iam)
        
#         inst_features = torch.cat(inst_features, -1)
#         iams = torch.cat(iams, dim=1)

#         if prev_inst_features is not None:
#             inst_features = torch.cat([inst_features, prev_inst_features], -1)
#             inst_features = F.relu_(self.fc_fuse(inst_features))
#             # inst_features = self.fc_fuse(inst_features)

#         inst_features = F.relu_(self.fc(inst_features))

#         # predict classification & segmentation kernel & objectness
#         pred_logits = self.cls_score(inst_features)
#         pred_kernel = self.mask_kernel(inst_features)
#         pred_scores = self.objectness(inst_features)
#         pred_bboxes = self.bbox_pred(inst_features)


#         results = {
#             'logits': pred_logits,
#             'mask_kernel': pred_kernel,
#             'border_kernel': pred_kernel,
#             'objectness_scores': pred_scores,
#             'bboxes': pred_bboxes,
#             'iam': iams,
#             'inst_feats': inst_features
#         }
        
#         return results


@HEADS.register(name="InstanceHead-v2.0-overlaps")
class InstanceHead(nn.Module):
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
        self.full_mask_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        self.overlap_mask_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        self.visible_mask_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        
        expand_dim = self.dim * self.num_groups
        self.fc_f = nn.Linear(expand_dim, expand_dim)
        self.fc_o = nn.Linear(expand_dim, expand_dim)
        self.fc_v = nn.Linear(expand_dim, expand_dim)
        self.fc_fuse = nn.Linear(expand_dim*2, expand_dim)

        # Outputs
        self.cls_score = nn.Linear(expand_dim, self.num_classes)
        self.full_mask_kernel = nn.Linear(expand_dim, self.kernel_dim)
        self.overlap_mask_kernel = nn.Linear(expand_dim, self.kernel_dim)
        self.visible_mask_kernel = nn.Linear(expand_dim, self.kernel_dim)
        self.objectness = nn.Linear(expand_dim, 1)
        self.bbox_pred = nn.Linear(expand_dim, 4)
        
        self.prior_prob = 0.01
        self._init_weights()
        
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))
        print("v2.0-overlaps")


    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, bias_value)

        init.normal_(self.full_mask_kernel.weight, std=0.01)
        init.constant_(self.full_mask_kernel.bias, 0.0)

        init.normal_(self.overlap_mask_kernel.weight, std=0.01)
        init.constant_(self.overlap_mask_kernel.bias, 0.0)

        init.normal_(self.visible_mask_kernel.weight, std=0.01)
        init.constant_(self.visible_mask_kernel.bias, 0.0)

        c2_xavier_fill(self.fc_f)
        c2_xavier_fill(self.fc_o)
        c2_xavier_fill(self.fc_v)
        c2_xavier_fill(self.fc_fuse)


    def forward(self, features, prev_inst_features=None):
        full_mask_iam = self.full_mask_iam(features)
        overlap_mask_iam = self.overlap_mask_iam(features)
        visible_mask_iam = self.visible_mask_iam(features)

        B, N, H, W = full_mask_iam.shape
        C = features.size(1)
        
        if self.activation == "softmax":
            full_mask_iam_prob = F.softmax(full_mask_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
            overlap_mask_iam_prob = F.softmax(overlap_mask_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
            visible_mask_iam_prob = F.softmax(visible_mask_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
        else:
            raise NotImplementedError(f"No activation {self.activation} found!")
        
        full_mask_inst_features = torch.bmm(full_mask_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        full_mask_inst_features = full_mask_inst_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        overlap_mask_inst_features = torch.bmm(overlap_mask_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        overlap_mask_inst_features = overlap_mask_inst_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        visible_mask_inst_features = torch.bmm(visible_mask_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        visible_mask_inst_features = visible_mask_inst_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)


        full_mask_inst_features = F.relu_(self.fc_f(full_mask_inst_features))
        overlap_mask_inst_features = F.relu_(self.fc_o(overlap_mask_inst_features))
        visible_mask_inst_features = F.relu_(self.fc_v(visible_mask_inst_features))


        # predictions.
        pred_logits = self.cls_score(full_mask_inst_features)
        
        pred_full_mask_kernel = self.full_mask_kernel(full_mask_inst_features)
        pred_overlap_mask_kernel = self.overlap_mask_kernel(overlap_mask_inst_features)
        pred_visible_mask_kernel = self.visible_mask_kernel(visible_mask_inst_features)

        pred_scores = self.objectness(full_mask_inst_features)
        pred_bboxes = self.bbox_pred(full_mask_inst_features)

        results = {
            'logits': pred_logits,
            'mask_kernel': pred_full_mask_kernel,
            'overlap_mask_kernel': pred_overlap_mask_kernel,
            'visible_mask_kernel': pred_visible_mask_kernel,
            'objectness_scores': pred_scores,
            'bboxes': pred_bboxes,
            'iam': full_mask_iam,
            'inst_feats': full_mask_inst_features
        }

        return results



if __name__ == "__main__":
    instance_head = InstanceBranch(in_channels=256, num_classes=1, kernel_dim=10, num_masks=5, num_groups=2)
    x = torch.rand(1, 256, 512, 512)
    pred_logits, pred_kernel, pred_scores, iam = instance_head(x)
    print(pred_kernel.shape)
    print(pred_logits.shape)