import torch 
from torch import digamma, nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F

# import math
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

import sys
sys.path.append('.')

from ..common import _make_stack_3x3_convs, MLP
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


@HEADS.register(name="OccluderInstanceBranch")
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
        self.iam_conv = nn.Conv2d(self.dim, self.num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)
        
        expand_dim = self.dim * self.num_groups
        self.fc = nn.Linear(expand_dim, expand_dim)

        # outputs
        self.cls_score = nn.Linear(expand_dim, self.num_classes)
        self.mask_kernel = nn.Linear(expand_dim, self.kernel_dim)
        self.objectness = nn.Linear(expand_dim, 1)
        
        self.prior_prob = 0.01
        self._init_weights()

        self.softmax_bias = nn.Parameter(torch.ones([1, ]))


    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.iam_conv, self.cls_score]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.iam_conv.weight, std=0.01)
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
        if self.activation == "softmax":
            iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)
        else:
            raise NotImplementedError(f"No activation {self.activation} found!")


        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(
            iam_prob, features.view(B, C, -1).permute(0, 2, 1))

        inst_features = inst_features.reshape(
            B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
        inst_features = F.relu_(self.fc(inst_features))

        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)

        return pred_logits, pred_kernel, pred_scores, iam
    




# class IAMClsHead(nn.Module):
#     def __init__(self, input_dim, output_dim, num_convs):
#         super(IAMClsHead, self).__init__()

#         self.bn = nn.BatchNorm2d(input_dim)
#         self.relu = nn.ReLU(inplace=True)
#         # self.iam_head = _make_stack_3x3_convs(num_convs, input_dim, output_dim)

#         self.cls_head = nn.Sequential(
#             nn.Conv2d(input_dim, input_dim, 3, padding=1),
#             nn.BatchNorm2d(input_dim),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(input_dim, output_dim, 3, padding=1),
#             nn.BatchNorm2d(output_dim),
#             nn.ReLU(inplace=True),
#         )

#         self.adaptive_pool = nn.AdaptiveAvgPool2d(32)
#         # self.fc = nn.Linear(1024, 1024)

#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.cls_head(x)
#         x = self.adaptive_pool(x)
#         x = x.view(x.size(0), x.size(1), -1)
#         # x = F.relu_(self.fc(x))

#         return x
    

# class InstanceBranch(nn.Module):
#     def __init__(self, dim, kernel_dim, num_masks=10):
#         super().__init__()
#         dim = dim
#         num_convs = 4
#         num_masks = num_masks
#         kernel_dim = kernel_dim
        
#         self.num_groups = 2
#         self.num_classes = 2
        
#         # iam prediction, a simple conv
#         self.iam_conv = nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)
#         # self.value_proj = _make_stack_3x3_convs(1, dim, dim)

#         expand_dim = dim 
#         self.fc1 = nn.Linear(expand_dim, expand_dim)
#         self.fc2 = nn.Linear(expand_dim, expand_dim)

#         # outputs
#         self.cls_score = nn.Linear(expand_dim, self.num_classes)
#         self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
#         self.occl_kernel = nn.Linear(expand_dim, kernel_dim)
#         self.objectness = nn.Linear(expand_dim, 1)
        
#         self.prior_prob = 0.01
#         self._init_weights()
        
#         self.softmax_bias = nn.Parameter(torch.ones([1, ]))
#         # self.temperature = nn.Parameter(torch.tensor([30.]))
#         # self.temperature = 30.0


#     def _init_weights(self):
#         bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
#         for module in [self.iam_conv, self.cls_score]:
#             init.constant_(module.bias, bias_value)
#         init.normal_(self.iam_conv.weight, std=0.01)
#         init.normal_(self.cls_score.weight, std=0.01)
#         init.normal_(self.mask_kernel.weight, std=0.01)
#         init.constant_(self.mask_kernel.bias, 0.0)
#         init.normal_(self.occl_kernel.weight, std=0.01)
#         init.constant_(self.occl_kernel.bias, 0.0)

#         c2_xavier_fill(self.fc1)
#         c2_xavier_fill(self.fc2)


#     def forward(self, features, idx=None):
#         # predict instance activation maps
#         iam = self.iam_conv(features)
#         # features = self.value_proj(features)
        
        
#         B, N, H, W = iam.shape
#         C = features.size(1)
        
#         # BxNxHxW -> BxNx(HW)
#         iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)
#         # iam_prob = F.softmax((iam.view(B, N, -1) + self.softmax_bias) / self.temperature, dim=-1)
#         # if self.temperature > 1:
#         #     self.temperature -= 0.5
#         # iam_prob = F.softmax(iam.view(B, N, -1), dim=-1)

#         # iam_prob = iam.sigmoid()
#         # iam_prob = iam_prob.view(B, N, -1)
#         # normalizer = iam_prob.sum(-1).clamp(min=1e-6)
#         # iam_prob = iam_prob / normalizer[:, :, None]

#         # aggregate features: BxCxHxW -> Bx(HW)xC
#         inst_features = torch.bmm(
#             iam_prob, features.view(B, C, -1).permute(0, 2, 1))

#         inst_features = inst_features.reshape(B, self.num_groups, N // self.num_groups, -1)
#         inst_features_mask = inst_features[:, 0, :, :]
#         inst_features_occl = inst_features[:, 1, :, :]

#         inst_features_mask = F.relu_(self.fc1(inst_features_mask))
#         inst_features_occl = F.relu_(self.fc2(inst_features_occl))

#         # predict classification & segmentation kernel & objectness
#         pred_logits = self.cls_score(inst_features_mask)
#         pred_mask_kernel = self.mask_kernel(inst_features_mask)
#         pred_occl_kernel = self.occl_kernel(inst_features_occl)
#         pred_scores = self.objectness(inst_features_mask)

#         return pred_logits, pred_mask_kernel, pred_occl_kernel, pred_scores, iam
    



if __name__ == "__main__":
    instance_head = InstanceBranch(dim=256, kernel_dim=10, num_masks=5)
    x = torch.rand(1, 256, 512, 512)
    # y = torch.rand(1, 5, 25, 25)
    # inst_logits, inst_kernel, inst_scores, inst_iam, occl_logits, occl_kernel, occl_scores, occl_iam = instance_head(x, y)
    # print(inst_logits.shape, occl_logits.shape)
    # print(inst_kernel.shape, occl_kernel.shape)
    # print(inst_iam.shape, occl_iam.shape)
    pred_logits, pred_mask_kernel, pred_occl_kernel, pred_scores, iam = instance_head(x)
    print(pred_mask_kernel.shape)
    print(pred_logits.shape)