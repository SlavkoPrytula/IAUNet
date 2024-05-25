import torch 
from torch import nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F

import math
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from models.seg.heads.common import _make_stack_3x3_convs
from models.seg.heads.instance_head.iam import IAM, DeepIAM

from configs import cfg
    
    

class PriorInstanceBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        dim = 256
        num_convs = 4
        
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
    

# class InstanceBranch(nn.Module):
#     def __init__(self, kernel_dim, num_masks=10):
#         super().__init__()
#         dim = 256
#         num_masks = num_masks
#         kernel_dim = kernel_dim
        
#         self.num_groups = 1
#         self.num_classes = 1
        
#         # iam prediction, a group conv
#         expand_dim = dim * self.num_groups
        
#         self.iam_conv = IAM(dim, num_masks * self.num_groups, self.num_groups)
        
#         # outputs
#         self.fc = nn.Linear(expand_dim, expand_dim)

#         self.cls_score = nn.Linear(expand_dim, self.num_classes)
#         self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
#         self.objectness = nn.Linear(expand_dim, 1)

#         self.prior_prob = 0.01
#         self._init_weights()

#     def _init_weights(self):
#         bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
#         for module in [self.cls_score]:
#             init.constant_(module.bias, bias_value)
#         init.normal_(self.cls_score.weight, std=0.01)

#         init.normal_(self.mask_kernel.weight, std=0.01)
#         init.constant_(self.mask_kernel.bias, 0.0)
#         c2_xavier_fill(self.fc)

#     def forward(self, features, idx=None):
#         # predict instance activation maps
#         iam = self.iam_conv(features)
#         # iam[:, :idx] += iam[:, 0].unsqueeze(1).clone()

#         iam_prob = iam.sigmoid()
#         # iam_prob[:, ]
#         # zero_mask = iam_prob[:, idx].unsqueeze(1) == 0
#         # iam_prob[:, :idx][zero_mask] = iam_prob[:, 0].unsqueeze(1)[zero_mask]
#         # print(torch.sum(zero_mask).item())

#         # iam_prob[:, :idx] = (iam_prob[:, :idx] + iam_prob[:, 0].unsqueeze(1)) / 2

#         B, N, H, W = iam_prob.shape
#         # temp_iam = iam_prob[:, 0].unsqueeze(1).clone()

#         # iam_prob[:, :idx, :-100, :] = 0
#         # iam_prob[:, :idx, :100, :] = temp_iam[:, :, :100, :] # (iam_prob[:, :idx, 10:, :] + iam_prob[:, 0, 10:, :].unsqueeze(1))

#         C = features.size(1)
        
#         # BxNxHxW -> BxNx(HW)
#         iam_prob = iam_prob.view(B, N, -1)
#         normalizer = iam_prob.sum(-1).clamp(min=1e-6)
#         iam_prob = iam_prob / normalizer[:, :, None]

#         iam_prob = iam_prob.view(B, N, H, W)
#         # iam_prob[:, :idx] = iam_prob[:, :idx] + iam_prob[:, 0].unsqueeze(1)
#         # temp_iam = iam_prob[:, 0].unsqueeze(1).clone()
#         # iam_prob[:, :idx, :-H//2, :] = 0
#         # iam_prob[:, :idx, :-H//2, :] = temp_iam[:, :, :-H//2, :]
#         iam_prob = iam_prob.view(B, N, -1)


#         # aggregate features: BxCxHxW -> Bx(HW)xC
#         inst_features = torch.bmm(
#             iam_prob, features.view(B, C, -1).permute(0, 2, 1))

#         inst_features = inst_features.reshape(
#             B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(
#             B, N // self.num_groups, -1)
        
#         inst_features = F.relu_(self.fc(inst_features))
        
#         # predict classification & segmentation kernel & objectness
#         pred_logits = self.cls_score(inst_features)
#         pred_kernel = self.mask_kernel(inst_features)
#         pred_scores = self.objectness(inst_features)
        
#         iam = iam_prob.view(B, N, H, W)
#         return pred_logits, pred_kernel, pred_scores, iam


# class InstanceBranch(nn.Module):
#     def __init__(self, kernel_dim=128, num_masks=10):
#         super().__init__()
#         dim = 256
#         num_convs = 4
#         num_masks = num_masks
#         kernel_dim = kernel_dim
#         self.num_classes = 1
        
#         # iam prediction, a simple conv
#         self.iam_conv = nn.Conv2d(dim, num_masks, 3, padding=1)

#         # outputs
#         self.cls_score = nn.Linear(dim, self.num_classes)
#         self.mask_kernel = nn.Linear(dim, kernel_dim)
#         self.objectness = nn.Linear(dim, 1)
        
#         self.prior_prob = 0.01
#         self._init_weights()

#     def _init_weights(self):
#         bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
#         for module in [self.iam_conv, self.cls_score]:
#             init.constant_(module.bias, bias_value)
#         init.normal_(self.cls_score.weight, std=0.01)

#         init.normal_(self.mask_kernel.weight, std=0.01)
#         init.constant_(self.mask_kernel.bias, 0.0)

#     def forward(self, features):
#         # predict instance maps
#         iam = self.iam_conv(features)
# #         iam_prob = F.softmax(torch.logsigmoid(iam.view(B, N, 1)), dim=-1)
#         iam_prob = nn.Sigmoid()(iam)      # might replace with predicting nuclues
        
#         B, N = iam_prob.shape[:2]   # (bs, num_masks)
#         C = features.size(1)        # dim = 256
        
#         # BxNxHxW -> BxNx(HW)
#         iam_prob = iam_prob.view(B, N, -1)   # (B, N, H, W) -> (B, N, [HW])
#         normalizer = iam_prob.sum(-1).clamp(min=1e-6)
#         iam_prob = iam_prob / normalizer[:, :, None]
        
#         features = features.view(B, C, -1).permute(0, 2, 1)  # (B, C, H, W) -> (B, [HW], C)

#         # aggregate features
#         inst_features = torch.matmul(
#             iam_prob,    # (B, N, [HW]) - instance masks proposals
#             features     # (B, [HW], C) - features from x4 convs
#         ) # -> (B, N, C) == (bs, num_masks, dim):: (2, 10, 256)
        
        
#         # predict classification & segmentation kernel & objectness
#         pred_logits = self.cls_score(inst_features)    # (B, N, C) -> (B, N, nc)
#         pred_kernel = self.mask_kernel(inst_features)  # (B, N, C) -> (B, N, 128)
#         pred_scores = self.objectness(inst_features)   # (B, N, C) -> (B, N, 1)
        
#         return pred_logits, pred_kernel, pred_scores, iam
    



# class InstanceBranch(nn.Module):
#     def __init__(self, kernel_dim, num_masks=10):
#         super().__init__()
#         dim = 256
#         num_masks = num_masks
#         kernel_dim = kernel_dim
        
#         self.num_groups = 1
#         self.num_classes = 1
        
#         # iam prediction, a group conv
#         expand_dim = dim * self.num_groups
        
#         self.iam_conv = nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)
        
#         # outputs
#         self.fc = nn.Linear(expand_dim, expand_dim)

#         self.cls_score = nn.Linear(expand_dim, self.num_classes)
#         self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
#         self.objectness = nn.Linear(expand_dim, 1)

#         self.prior_prob = 0.01
#         self._init_weights()


#     def _init_weights(self):
#         bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
#         for module in [self.iam_conv, self.cls_score]:
#             init.constant_(module.bias, bias_value)
#         init.normal_(self.iam_conv.weight, std=0.01)
#         init.normal_(self.cls_score.weight, std=0.01)
#         init.normal_(self.mask_kernel.weight, std=0.01)
#         init.constant_(self.mask_kernel.bias, 0.0)
        

#     def forward(self, features, idx=None):
#         # predict instance activation maps
#         iam = self.iam_conv(features)

#         iam_prob = iam.sigmoid()

#         B, N, H, W = iam_prob.shape
#         C = features.size(1)
        
#         # BxNxHxW -> BxNx(HW)
#         iam_prob = iam_prob.view(B, N, -1)
#         normalizer = iam_prob.sum(-1).clamp(min=1e-6)
#         iam_prob = iam_prob / normalizer[:, :, None]

#         # aggregate features: BxCxHxW -> Bx(HW)xC
#         inst_features = torch.bmm(
#             iam_prob, features.view(B, C, -1).permute(0, 2, 1))

#         inst_features = inst_features.reshape(
#             B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(
#             B, N // self.num_groups, -1)
        
#         inst_features = F.relu_(self.fc(inst_features))
        
#         # predict classification & segmentation kernel & objectness
#         pred_logits = self.cls_score(inst_features)
#         pred_kernel = self.mask_kernel(inst_features)
#         pred_scores = self.objectness(inst_features)
        
#         # iam = iam_prob.view(B, N, H, W)
#         return pred_logits, pred_kernel, pred_scores, iam    


class InstanceBranch(nn.Module):
    def __init__(self, kernel_dim, num_masks=10):
        super().__init__()
        dim = 256
        num_convs = 4
        num_masks = num_masks
        kernel_dim = kernel_dim
        
        self.num_groups = 1
        self.num_classes = 1
        
        # iam prediction, a simple conv
        self.iam_conv = nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)
        # self.iam_conv = DeepIAM(dim, num_masks * self.num_groups, self.num_groups)
        
        expand_dim = dim * self.num_groups
        self.fc = nn.Linear(expand_dim, expand_dim)

        # outputs
        self.cls_score = nn.Linear(expand_dim, self.num_classes)
        self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
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


    def forward(self, features, idx=None):
        # predict instance activation maps
        iam = self.iam_conv(features)

        B, N, H, W = iam.shape
        C = features.size(1)
        
        # BxNxHxW -> BxNx(HW)
        iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)
        # iam_prob = F.softmax(iam.view(B, N, -1), dim=-1)


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

        # iam = iam_prob.view(B, N, H, W)

        iam = {
            "iam": iam,
        }

        return pred_logits, pred_kernel, pred_scores, iam



# class GroupInstanceBranch(nn.Module):
#     def __init__(self, kernel_dim, num_masks=10):
#         super().__init__()
#         dim = 256
#         num_convs = 4
#         num_masks = num_masks
#         kernel_dim = kernel_dim
        
#         self.num_groups = 1
#         self.num_classes = 1
        
#         # iam prediction, a simple conv
#         self.iam_conv_0 = nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)
#         self.iam_conv_1 = nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)
        
#         expand_dim = dim * self.num_groups
#         self.fc = nn.Linear(expand_dim, expand_dim)

#         # outputs
#         self.cls_score = nn.Linear(expand_dim, self.num_classes)
#         self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
#         self.objectness = nn.Linear(expand_dim, 1)
        
#         self.prior_prob = 0.01
#         self._init_weights()
        
#         self.softmax_bias = nn.Parameter(torch.ones([1, ]))

#     def _init_weights(self):
#         bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
#         for module in [self.iam_conv_0, self.iam_conv_1, self.cls_score]:
#             init.constant_(module.bias, bias_value)
#         init.normal_(self.iam_conv_0.weight, std=0.01)
#         init.normal_(self.iam_conv_1.weight, std=0.01)
#         init.normal_(self.cls_score.weight, std=0.01)
#         init.normal_(self.mask_kernel.weight, std=0.01)
#         init.constant_(self.mask_kernel.bias, 0.0)


#     def forward(self, inst_feats, ovlp_feats, idx=None):
#         # 
#         # Xi --> IAM_0 --\
#         #                + --> IAM
#         # Xo --> IAM_1 --/
#         # 

#         # predict instance activation maps
#         inst_iam = self.iam_conv_0(inst_feats)
#         ovlp_iam = self.iam_conv_1(ovlp_feats)
#         # iam = inst_iam + ovlp_iam

#         B, N, H, W = iam.shape
#         C = inst_feats.size(1)
        
#         # BxNxHxW -> BxNx(HW)
#         iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)
#         # iam_prob = F.softmax(iam.view(B, N, -1), dim=-1)


#         # aggregate features: BxCxHxW -> Bx(HW)xC
#         inst_features = torch.bmm(
#             iam_prob, inst_feats.view(B, C, -1).permute(0, 2, 1))

#         inst_features = inst_features.reshape(
#             B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
#         inst_features = F.relu_(self.fc(inst_features))

#         # predict classification & segmentation kernel & objectness
#         pred_logits = self.cls_score(inst_features)
#         pred_kernel = self.mask_kernel(inst_features)
#         pred_scores = self.objectness(inst_features)

#         # iam = iam_prob.view(B, N, H, W)

#         iam = {
#             "iam": iam,
#             "instance_iam": inst_iam,
#             "overlap_iam": ovlp_iam
#         }

#         return pred_logits, pred_kernel, pred_scores, iam




class GroupInstanceBranch(nn.Module):
    def __init__(self, kernel_dim, num_masks=10):
        super().__init__()
        dim = 256
        num_convs = 4
        num_masks = num_masks
        kernel_dim = kernel_dim
        
        self.num_groups = 2
        self.num_classes = 1
        
        # iam prediction, a simple conv
        self.iam_conv = nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1)
        self.iam_conv_1 = nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1)
        
        expand_dim = dim * self.num_groups
        self.fc = nn.Linear(expand_dim, expand_dim)

        # outputs
        self.cls_score = nn.Linear(expand_dim, self.num_classes)
        self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
        self.objectness = nn.Linear(expand_dim, 1)
        
        self.prior_prob = 0.01
        self._init_weights()
        
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))

    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.iam_conv, self.iam_conv_1, self.cls_score]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.iam_conv.weight, std=0.01)
        init.normal_(self.iam_conv_1.weight, std=0.01)
        init.normal_(self.cls_score.weight, std=0.01)
        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)


    def forward(self, inst_feats, ovlp_feats, idx=None):
        # 
        # Xi --> IAM_0 --\
        #                + --> IAM
        # Xo --> IAM_1 --/
        # 
        
        # predict instance activation maps
        inst_iam = self.iam_conv(inst_feats)
        ovlp_iam = self.iam_conv_1(ovlp_feats)
        iam = torch.cat([inst_iam, ovlp_iam], dim=1)
        # iam = inst_iam + ovlp_iam

        # inst_feats = torch.cat([inst_feats, ovlp_feats], dim=1)

        B, N, H, W = iam.shape
        C = inst_feats.size(1)
        
        # BxNxHxW -> BxNx(HW)
        iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)

        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(
            iam_prob, inst_feats.view(B, C, -1).permute(0, 2, 1))

        inst_features = inst_features.reshape(
            B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
        inst_features = F.relu_(self.fc(inst_features))

        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)

        # iam = iam_prob.view(B, N, H, W)

        iam = {
            "iam": iam,
            "instance_iam": inst_iam,
            "overlap_iam": ovlp_iam
        }

        return pred_logits, pred_kernel, pred_scores, iam