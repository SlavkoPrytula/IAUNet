import torch 
from torch import nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

import sys
sys.path.append('.')

from .iam import (IAM)
from ..common import _make_stack_3x3_convs
from models.seg.nn.blocks import MLP
from utils.registry import HEADS


@HEADS.register(name="InstanceHead-v2.1-overlaps")
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
        
        hidden_dim = self.dim * self.num_groups
        self.fc_f = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_fuse = nn.Linear(hidden_dim*2, hidden_dim)

        self.mixer = MLP(hidden_dim*3, hidden_dim*6, hidden_dim, 2)

        # Outputs
        self.cls_score = nn.Linear(hidden_dim, self.num_classes)
        self.full_mask_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.overlap_mask_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.visible_mask_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.objectness = nn.Linear(hidden_dim, 1)
        self.bbox_pred = nn.Linear(hidden_dim, 4)
        
        self.prior_prob = 0.01
        self._init_weights()
        
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))
        print("v2.1-overlaps")


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

        # inst feats refining.
        inst_features = torch.cat([full_mask_inst_features, 
                                   overlap_mask_inst_features, 
                                   visible_mask_inst_features], dim=-1)

        full_mask_inst_features = self.mixer(inst_features)


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




@HEADS.register(name="InstanceHead-v2.2-overlaps")
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


        # branches.
        self.instance_branch = _make_stack_3x3_convs(
            in_channels=self.dim, 
            out_channels=self.dim, 
            num_convs=3
            )
        
        self.overlap_branch = _make_stack_3x3_convs(
            in_channels=self.dim * 2, 
            out_channels=self.dim, 
            num_convs=3
            )
        
        self.visible_branch = _make_stack_3x3_convs(
            in_channels=self.dim * 2, 
            out_channels=self.dim, 
            num_convs=3
            )

        self.r_instance_branch = _make_stack_3x3_convs(
            in_channels=self.dim * 3, 
            out_channels=self.dim, 
            num_convs=3
            )
        
        # iam.
        self.inst_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        self.overlap_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        self.visible_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        
        hidden_dim = self.dim * self.num_groups
        self.fc_f = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_fuse = nn.Linear(hidden_dim*2, hidden_dim)

        # outputs.
        self.cls_score = nn.Linear(hidden_dim, self.num_classes)
        self.inst_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.overlap_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.visible_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.objectness = nn.Linear(hidden_dim, 1)
        self.bbox_pred = nn.Linear(hidden_dim, 4)
        
        self.prior_prob = 0.01
        self._init_weights()
        
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))
        print("v2.2-overlaps")

    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, bias_value)

        init.normal_(self.inst_kernel.weight, std=0.01)
        init.constant_(self.inst_kernel.bias, 0.0)

        init.normal_(self.overlap_kernel.weight, std=0.01)
        init.constant_(self.overlap_kernel.bias, 0.0)

        init.normal_(self.visible_kernel.weight, std=0.01)
        init.constant_(self.visible_kernel.bias, 0.0)

        c2_xavier_fill(self.fc_f)
        c2_xavier_fill(self.fc_o)
        c2_xavier_fill(self.fc_v)
        c2_xavier_fill(self.fc_fuse)

        for m in self.instance_branch.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

        for m in self.overlap_branch.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

        for m in self.visible_branch.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

        for m in self.r_instance_branch.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
                

    def forward(self, features, prev_inst_features=None):
        inst_features = self.instance_branch(features)
        f_i = torch.cat([inst_features, features], dim=1)
        overlap_features = self.overlap_branch(f_i)
        visible_features = self.visible_branch(f_i)

        f_r = torch.cat([inst_features, overlap_features, visible_features], dim=1)
        inst_features = self.r_instance_branch(f_r)


        inst_iam = self.inst_iam(inst_features)
        overlap_iam = self.overlap_iam(overlap_features)
        visible_iam = self.visible_iam(visible_features)

        B, N, H, W = inst_iam.shape
        C = inst_features.size(1)
        
        if self.activation == "softmax":
            inst_iam_prob = F.softmax(inst_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
            overlap_iam_prob = F.softmax(overlap_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
            visible_iam_prob = F.softmax(visible_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
        else:
            raise NotImplementedError(f"No activation {self.activation} found!")
        
        inst_features = torch.bmm(inst_iam_prob, inst_features.view(B, C, -1).permute(0, 2, 1))
        inst_features = inst_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        overlap_features = torch.bmm(overlap_iam_prob, overlap_features.view(B, C, -1).permute(0, 2, 1))
        overlap_features = overlap_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        visible_features = torch.bmm(visible_iam_prob, visible_features.view(B, C, -1).permute(0, 2, 1))
        visible_features = visible_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        inst_features = F.relu_(self.fc_f(inst_features))
        overlap_features = F.relu_(self.fc_o(overlap_features))
        visible_features = F.relu_(self.fc_v(visible_features))


        # predictions.
        pred_logits = self.cls_score(inst_features)
        
        pred_inst_kernel = self.inst_kernel(inst_features)
        pred_overlap_kernel = self.overlap_kernel(overlap_features)
        pred_visible_kernel = self.visible_kernel(visible_features)

        pred_scores = self.objectness(inst_features)
        pred_bboxes = self.bbox_pred(inst_features)


        results = {
            'logits': pred_logits,
            'objectness_scores': pred_scores,
            'kernels': {
                'instance_kernel': pred_inst_kernel,
                'overlap_kernel': pred_overlap_kernel,
                'visible_kernel': pred_visible_kernel
                },
            'bboxes': {
                'instance_bboxes': pred_bboxes
                },
            'iams': {
                'instance_iams': inst_iam,
                'overlap_iams': overlap_iam,
                'visible_iams': visible_iam
                },
            'inst_feats': {
                'instance_feats': inst_features
                }
        }

        return results
    




@HEADS.register(name="InstanceHead-v2.3-overlaps")
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


        # branches.
        self.instance_branch = _make_stack_3x3_convs(
            in_channels=self.dim, 
            out_channels=self.dim, 
            num_convs=3
            )
        
        self.overlap_branch = _make_stack_3x3_convs(
            in_channels=self.dim * 2, 
            out_channels=self.dim, 
            num_convs=3
            )
        
        self.visible_branch = _make_stack_3x3_convs(
            in_channels=self.dim * 2, 
            out_channels=self.dim, 
            num_convs=3
            )

        self.r_instance_branch = _make_stack_3x3_convs(
            in_channels=self.dim * 3, 
            out_channels=self.dim, 
            num_convs=3
            )
        
        # iam.
        self.inst_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        self.overlap_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        self.visible_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        
        hidden_dim = self.dim * self.num_groups
        self.fc_f = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_fuse = nn.Linear(hidden_dim*2, hidden_dim)

        # outputs.
        self.cls_score = nn.Linear(hidden_dim, self.num_classes)
        self.inst_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.overlap_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.visible_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.objectness = nn.Linear(hidden_dim, 1)
        self.bbox_pred = nn.Linear(hidden_dim, 4)
        
        self.prior_prob = 0.01
        self._init_weights()
        
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))
        print("v2.3-overlaps")

    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, bias_value)

        init.normal_(self.inst_kernel.weight, std=0.01)
        init.constant_(self.inst_kernel.bias, 0.0)

        init.normal_(self.overlap_kernel.weight, std=0.01)
        init.constant_(self.overlap_kernel.bias, 0.0)

        init.normal_(self.visible_kernel.weight, std=0.01)
        init.constant_(self.visible_kernel.bias, 0.0)

        c2_xavier_fill(self.fc_f)
        c2_xavier_fill(self.fc_o)
        c2_xavier_fill(self.fc_v)
        c2_xavier_fill(self.fc_fuse)

        for m in self.instance_branch.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

        for m in self.overlap_branch.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

        for m in self.visible_branch.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

        for m in self.r_instance_branch.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
                

    def forward(self, features, prev_inst_features=None):
        inst_features = self.instance_branch(features)
        f_i = torch.cat([inst_features, features], dim=1)
        overlap_features = self.overlap_branch(f_i)
        visible_features = self.visible_branch(f_i)

        f_r = torch.cat([inst_features, overlap_features, visible_features], dim=1)
        inst_features = self.r_instance_branch(f_r)


        inst_iam = self.inst_iam(inst_features)
        overlap_iam = self.overlap_iam(overlap_features)
        visible_iam = self.visible_iam(visible_features)

        B, N, H, W = inst_iam.shape
        C = inst_features.size(1)
        
        if self.activation == "softmax":
            inst_iam_prob = F.softmax(inst_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
            overlap_iam_prob = F.softmax(overlap_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
            visible_iam_prob = F.softmax(visible_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
        else:
            raise NotImplementedError(f"No activation {self.activation} found!")
        
        inst_features = torch.bmm(inst_iam_prob, inst_features.view(B, C, -1).permute(0, 2, 1))
        inst_features = inst_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        overlap_features = torch.bmm(overlap_iam_prob, overlap_features.view(B, C, -1).permute(0, 2, 1))
        overlap_features = overlap_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        visible_features = torch.bmm(visible_iam_prob, visible_features.view(B, C, -1).permute(0, 2, 1))
        visible_features = visible_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        inst_features = F.relu_(self.fc_f(inst_features))
        overlap_features = F.relu_(self.fc_o(overlap_features))
        visible_features = F.relu_(self.fc_v(visible_features))


        # predictions.
        pred_logits = self.cls_score(inst_features)
        
        pred_inst_kernel = self.inst_kernel(inst_features)
        pred_overlap_kernel = self.overlap_kernel(overlap_features)
        pred_visible_kernel = self.visible_kernel(visible_features)

        pred_scores = self.objectness(inst_features)
        pred_bboxes = self.bbox_pred(inst_features)


        results = {
            'logits': pred_logits,
            'objectness_scores': pred_scores,
            'kernels': {
                'instance_kernel': pred_inst_kernel,
                'overlap_kernel': pred_overlap_kernel,
                'visible_kernel': pred_visible_kernel
                },
            'bboxes': {
                'instance_bboxes': pred_bboxes
                },
            'iams': {
                'instance_iams': inst_iam,
                'overlap_iams': overlap_iam,
                'visible_iams': visible_iam
                },
            'inst_feats': {
                'instance_feats': inst_features
                }
        }

        return results