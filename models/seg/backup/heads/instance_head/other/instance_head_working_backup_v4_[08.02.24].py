import torch 
from torch import digamma, nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F

# import math
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

import sys
sys.path.append('.')

# from models.seg.blocks import ContextBlock
from models.seg.heads.common import _make_stack_3x3_convs
# from models.seg.heads.common import Block
# from models.seg.heads.instance_head.iam import IAM, DeepIAM

from configs import cfg
    
    

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
    


# class InstanceBranch(nn.Module):
#     def __init__(self, dim, kernel_dim, num_masks=10):
#         super().__init__()
#         dim = dim
#         num_convs = 4
#         num_masks = num_masks
#         kernel_dim = kernel_dim
        
#         self.num_groups = 1
#         self.num_classes = 2
        
#         self.iam_conv = nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)

#         # self.iam_head1 = _make_stack_3x3_convs(num_convs, num_masks, num_masks)
        
#         expand_dim = dim * self.num_groups
#         self.fc = nn.Linear(expand_dim, expand_dim)

#         # outputs
#         # self.cls_head = nn.Sequential(
#         #     nn.Conv2d(dim, dim, 3, padding=1),
#         #     nn.BatchNorm2d(dim),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(dim, num_masks, 3, padding=1),
#         #     nn.BatchNorm2d(num_masks),
#         #     nn.ReLU(inplace=True),
#         # )
#         # self.cls_head = _make_stack_3x3_convs(2, dim, dim)
#         # self.cls_head = ConvFCCLSHead(dim, inner_dim=256, out_dim=num_masks, num_classes=self.num_classes)
#         # self.cls_score = nn.Linear(1024, self.num_classes)
        
#         # self.fc_cls = nn.Linear(1024, 1024)

#         self.cls_score = nn.Linear(expand_dim, self.num_classes)
#         self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
#         self.objectness = nn.Linear(expand_dim, 1)
        
#         self.prior_prob = 0.01
#         self.softmax_bias = nn.Parameter(torch.ones([1, ]))

#         self._init_weights()


#     def _init_weights(self):
#         bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
#         for module in [self.iam_conv, self.cls_score]:
#             init.constant_(module.bias, bias_value)
#         init.normal_(self.iam_conv.weight, std=0.01)
#         init.normal_(self.cls_score.weight, std=0.01)
#         init.normal_(self.mask_kernel.weight, std=0.01)
#         init.constant_(self.mask_kernel.bias, 0.0)

#         c2_xavier_fill(self.fc)
        

#     def forward(self, features, idx=None):
#         # predict instance activation maps
#         iam = self.iam_conv(features)

#         B, N, H, W = iam.shape
#         C = features.size(1)
        
#         # BxNxHxW -> BxNx(HW)
#         # iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)

#         iam_prob = iam.sigmoid()
#         iam_prob = iam_prob.view(B, N, -1)
#         normalizer = iam_prob.sum(-1).clamp(min=1e-9)
#         iam_prob = iam_prob / normalizer[:, :, None]


#         # aggregate features: BxCxHxW -> Bx(HW)xC
#         inst_features = torch.bmm(
#             iam_prob, 
#             features.view(B, C, -1).permute(0, 2, 1)
#             )

#         inst_features = inst_features.reshape(
#             B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
#         inst_features = F.relu_(self.fc(inst_features))

#         # predict classification & segmentation kernel & objectness
#         pred_logits = self.cls_score(inst_features)
#         pred_kernel = self.mask_kernel(inst_features)
#         pred_scores = self.objectness(inst_features)

#         return pred_logits, pred_kernel, pred_scores, iam




# class InstanceBranch(nn.Module):
#     def __init__(self, dim, kernel_dim, num_masks=10):
#         super().__init__()
#         dim = dim
#         num_convs = 4
#         num_masks = num_masks
#         kernel_dim = kernel_dim
        
#         self.num_groups = 1
#         self.num_classes = 2
        
#         self.iam_conv = nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)

#         expand_dim = dim * self.num_groups

#         # Classification branch
#         self.cls_branch = nn.Sequential(
#             nn.Linear(expand_dim, expand_dim),
#             nn.ReLU(),
#             nn.Linear(expand_dim, expand_dim),
#             nn.ReLU(),
#             nn.Linear(expand_dim, self.num_classes)
#         )

#         # Kernel branch
#         self.kernel_branch = nn.Sequential(
#             nn.Linear(expand_dim, expand_dim),
#             nn.ReLU(),
#             nn.Linear(expand_dim, expand_dim),
#             nn.ReLU(),
#             nn.Linear(expand_dim, kernel_dim)
#         )

#         # Objectness branch
#         self.objectness_branch = nn.Sequential(
#             nn.Linear(expand_dim, expand_dim),
#             nn.ReLU(),
#             nn.Linear(expand_dim, 1)
#         )

#         self.prior_prob = 0.01
#         self.softmax_bias = nn.Parameter(torch.ones([1, ]))
#         self._init_weights()


#     def _init_weights(self):
#         bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
#         for module in [self.iam_conv, self.cls_branch[-1], self.kernel_branch[-1], self.objectness_branch[-1]]:
#             init.constant_(module.bias, bias_value)
#         init.normal_(self.iam_conv.weight, std=0.01)
#         init.normal_(self.cls_branch[-1].weight, std=0.01)
#         init.normal_(self.kernel_branch[-1].weight, std=0.01)
#         init.normal_(self.objectness_branch[-1].weight, std=0.01)
        

#     def forward(self, features, idx=None):
#         # predict instance activation maps
#         iam = self.iam_conv(features)

#         B, N, H, W = iam.shape
#         C = features.size(1)
        
#         # BxNxHxW -> BxNx(HW)
#         iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)

#         # iam_prob = iam.sigmoid()
#         # iam_prob = iam_prob.view(B, N, -1)
#         # normalizer = iam_prob.sum(-1).clamp(min=1e-9)
#         # iam_prob = iam_prob / normalizer[:, :, None]


#         inst_features = torch.bmm(iam_prob, features.view(B, C, -1).permute(0, 2, 1))
#         inst_features = inst_features.view(B, N, -1)

#         pred_logits = self.cls_branch(inst_features)
#         pred_kernel = self.kernel_branch(inst_features)
#         pred_scores = self.objectness_branch(inst_features)

#         return pred_logits, pred_kernel, pred_scores, iam



# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, drop=0):
#         super().__init__()
#         hidden_features = hidden_features or in_features
#         out_features = out_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = nn.GELU()
#         self.drop1 = nn.Dropout(drop)
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop2 = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop1(x)
#         x = self.fc2(x)
#         x = self.drop2(x)
#         return x
    

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# class InstanceBranch(nn.Module):
#     def __init__(self, dim, kernel_dim, num_masks=10):
#         super().__init__()
#         dim = dim
#         num_convs = 4
#         num_masks = num_masks
#         kernel_dim = kernel_dim
        
#         self.num_groups = 1
#         self.num_classes = 2
        
#         # iam prediction, a simple conv
#         self.iam_conv = nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)
#         self.value_proj = _make_stack_3x3_convs(1, dim, dim)

#         # # self.iam_conv = DeepIAM(dim, num_masks * self.num_groups, self.num_groups)
#         self.bn1 = nn.BatchNorm2d(num_masks)
#         self.relu1 = nn.ReLU(inplace=True)
#         # self.bn2 = nn.BatchNorm2d(num_masks)
#         # self.relu2 = nn.ReLU(inplace=True)

#         # self.cls_head = _make_stack_3x3_convs(num_convs, num_masks, num_masks)
#         # self.iou_head = _make_stack_3x3_convs(num_convs, num_masks, num_masks)
#         # self.iam_head2 = _make_stack_3x3_convs(num_convs, num_masks, num_masks)
        
#         expand_dim = dim * self.num_groups
#         # self.fc = nn.Linear(expand_dim, expand_dim)
#         # self.norm1 = nn.LayerNorm(expand_dim)
#         # self.norm2 = nn.LayerNorm(expand_dim)
#         # self.ffn = Mlp(expand_dim, expand_dim, expand_dim)

#         # outputs
#         self.cls_head = nn.Sequential(
#             nn.Conv2d(dim, dim, 3, padding=1),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim, num_masks, 3, padding=1),
#             nn.BatchNorm2d(num_masks),
#             nn.ReLU(inplace=True),
#         )
#         # self.cls_head = _make_stack_3x3_convs(2, dim, dim)
#         # self.cls_head = ConvFCCLSHead(dim, inner_dim=256, out_dim=num_masks, num_classes=self.num_classes)
#         # self.cls_score = nn.Linear(1024, self.num_classes)
        
#         # self.fc_cls = nn.Linear(1024, 1024)

#         self.cls_score = nn.Linear(expand_dim, self.num_classes)
#         # self.cls_score = nn.Linear(1024, self.num_classes)
#         self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
#         self.objectness = nn.Linear(expand_dim, 1)
        
#         self.prior_prob = 0.01
#         self._init_weights()
        
#         self.softmax_bias = nn.Parameter(torch.ones([1, ]))
#         # self.temperature = nn.Parameter(torch.tensor([30.]))
#         # self.temperature = 30.0


#     def _init_weights(self):
#         bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
#         for module in [self.iam_conv]:
#             init.constant_(module.bias, bias_value)
#         init.normal_(self.iam_conv.weight, std=0.01)
#         init.normal_(self.cls_score.weight, std=0.01)
#         init.normal_(self.mask_kernel.weight, std=0.01)
#         init.constant_(self.mask_kernel.bias, 0.0)

#         # c2_xavier_fill(self.fc)

#         # for m in self.value_proj.modules():
#         #     if isinstance(m, nn.Conv2d):
#         #         c2_msra_fill(m)
        
#         # for m in self.cls_head.modules():
#         #     if isinstance(m, nn.Conv2d):
#         #         c2_msra_fill(m)
        
#         # for m in self.iou_head.modules():
#         #     if isinstance(m, nn.Conv2d):
#         #         c2_msra_fill(m)


#     def forward(self, features, idx=None):
#         # predict instance activation maps
#         iam = self.iam_conv(features)
#         # features = self.value_proj(features)

#         # x_cls = self.cls_head(features)
#         # x_cls = nn.AvgPool2d(16)(x_cls)
#         # x_cls = x_cls.view(x_cls.size(0), x_cls.size(1), -1)
        
#         # x_cls = self.cls_score(x_cls)
#         # print(x_cls.shape)

#         # x = iam
#         # x = self.bn1(x)
#         # x = self.relu1(x)
#         # x = self.cls_head(x)
#         # x = nn.AvgPool2d(16)(x)
#         # x = x.view(x.size(0), x.size(1), -1)
#         # x = F.relu_(self.fc_cls(x))

#         # x_cls = self.cls_head(features)
        

#         B, N, H, W = iam.shape
#         C = features.size(1)
        
#         # BxNxHxW -> BxNx(HW)
#         # print(torch.max(iam.view(B, N, -1) + self.softmax_bias), torch.max((iam.view(B, N, -1) + self.softmax_bias) / self.temperature))
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

#         inst_features = inst_features.reshape(
#             B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        

#         # inst_features = self.norm1(inst_features)
#         # inst_features = self.ffn(inst_features)
#         # inst_features = inst_features + _inst_features
#         # inst_features = self.norm2(inst_features)

#         # inst_features = F.relu_(self.fc(inst_features))

#         # predict classification & segmentation kernel & objectness
#         pred_logits = self.cls_score(inst_features)
#         # pred_logits = self.cls_score(x_cls)
#         # pred_logits = self.cls_score(x)
#         pred_kernel = self.mask_kernel(inst_features)
#         pred_scores = self.objectness(inst_features)

#         return pred_logits, pred_kernel, pred_scores, iam
    



# class IAMClsHead(nn.Module):
#     def __init__(self, dim, num_convs):
#         super().__init__()
#         self.bn = nn.BatchNorm2d(dim)
#         self.relu = nn.ReLU(inplace=True)
#         self.iam_head = _make_stack_3x3_convs(num_convs, dim, dim)

#         self._init_weights()

#     def _init_weights(self):
#         for m in self.iam_head.modules():
#             if isinstance(m, nn.Conv2d):
#                 c2_msra_fill(m)

#     def forward(self, x):
#         # x are input features from conv2d layer (not normalized)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.iam_head(x)
#         x = nn.AvgPool2d(16)(x)
#         x = x.view(x.size(0), x.size(1), -1)

#         return x


class IAMClsHead(nn.Module):
    def __init__(self, input_dim, output_dim, num_convs):
        super(IAMClsHead, self).__init__()

        self.bn = nn.BatchNorm2d(input_dim)
        self.relu = nn.ReLU(inplace=True)
        # self.iam_head = _make_stack_3x3_convs(num_convs, input_dim, output_dim)

        self.cls_head = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, output_dim, 3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d(32)
        # self.fc = nn.Linear(1024, 1024)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.cls_head(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), x.size(1), -1)
        # x = F.relu_(self.fc(x))

        return x



# class InstanceBranch(nn.Module):
#     def __init__(self, dim, kernel_dim, num_masks=10):
#         super().__init__()
#         dim = dim
#         num_convs = 4
#         num_masks = num_masks
#         kernel_dim = kernel_dim
        
#         self.num_groups = 1
#         self.num_classes = 2

#         expand_dim = dim * self.num_groups
        
#         # iam prediction, a simple conv
#         self.iam_conv = nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)

#         self.cls_head = IAMClsHead(dim, num_masks, num_convs=2)
        
#         self.fc = nn.Linear(expand_dim, expand_dim)
#         # self.ffn = MLP(expand_dim, expand_dim, expand_dim, num_layers=2)

#         # outputs
#         self.cls_score = nn.Linear(1024, self.num_classes)
#         self.mask_kernel = MLP(expand_dim, expand_dim, kernel_dim, num_layers=3)
#         # self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
#         self.objectness = nn.Linear(expand_dim, 1)
        
#         self.prior_prob = 0.01
#         self._init_weights()
        
#         self.softmax_bias = nn.Parameter(torch.ones([1, ]))


#     def _init_weights(self):
#         bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
#         for module in [self.iam_conv]:
#             init.constant_(module.bias, bias_value)
#         init.normal_(self.iam_conv.weight, std=0.01)
#         init.normal_(self.cls_score.weight, std=0.01)
#         # init.normal_(self.mask_kernel.weight, std=0.01)
#         # init.constant_(self.mask_kernel.bias, 0.0)


#     def forward(self, features, idx=None):
#         # predict instance activation maps
#         iam = self.iam_conv(features)

#         x_cls = self.cls_head(features)

#         B, N, H, W = iam.shape
#         C = features.size(1)
        
#         # BxNxHxW -> BxNx(HW)
#         iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)

#         # aggregate features: BxCxHxW -> Bx(HW)xC
#         inst_features = torch.bmm(
#             iam_prob, features.view(B, C, -1).permute(0, 2, 1))

#         # inst_features = inst_features.reshape(
#         #     B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
#         inst_features = F.relu_(self.fc(inst_features))
#         # inst_features = self.ffn(inst_features)

#         # predict classification & segmentation kernel & objectness
#         pred_logits = self.cls_score(x_cls)
#         pred_kernel = self.mask_kernel(inst_features)
#         pred_scores = self.objectness(inst_features)

#         return pred_logits, pred_kernel, pred_scores, iam
    

class InstanceBranch(nn.Module):
    def __init__(self, dim, kernel_dim, num_masks=10):
        super().__init__()
        dim = dim
        num_convs = 4
        num_masks = num_masks
        kernel_dim = kernel_dim
        
        self.num_groups = 1
        self.num_classes = 2
        
        # iam prediction, a simple conv
        self.iam_conv = nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)
        # # self.iam_conv = DeepIAM(dim, num_masks * self.num_groups, self.num_groups)
        self.bn1 = nn.BatchNorm2d(num_masks)
        self.relu1 = nn.ReLU(inplace=True)
        # self.bn2 = nn.BatchNorm2d(num_masks)
        # self.relu2 = nn.ReLU(inplace=True)

        # self.iam_head1 = _make_stack_3x3_convs(num_convs, num_masks, num_masks)
        # self.iam_head2 = _make_stack_3x3_convs(num_convs, num_masks, num_masks)
        
        expand_dim = dim * self.num_groups
        self.fc = nn.Linear(expand_dim, expand_dim)

        # outputs
        # self.cls_head = ConvFCCLSHead(dim, inner_dim=256, out_dim=num_masks, num_classes=self.num_classes)
        # self.cls_score = nn.Linear(1024, self.num_classes)
        
        # self.fc_cls = nn.Linear(1024, 1024)

        self.cls_score = nn.Linear(expand_dim, self.num_classes)
        self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
        self.objectness = nn.Linear(expand_dim, 1)
        
        self.prior_prob = 0.01
        self._init_weights()
        
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))
        # self.temperature = nn.Parameter(torch.tensor([30.]))
        # self.temperature = 30.0


    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.iam_conv, self.cls_score]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.iam_conv.weight, std=0.01)
        init.normal_(self.cls_score.weight, std=0.01)
        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)

        c2_xavier_fill(self.fc)

        # for m in self.iam_head1.modules():
        #     if isinstance(m, nn.Conv2d):
        #         c2_msra_fill(m)

        # for m in self.iam_head2.modules():
        #     if isinstance(m, nn.Conv2d):
        #         c2_msra_fill(m)


    def forward(self, features, idx=None):
        # predict instance activation maps
        iam = self.iam_conv(features)

        # x = iam
        # x = self.bn1(x)
        # x = self.relu1(x)
        # x = self.iam_head1(x)
        # # x = nn.AvgPool2d(4)(x)
        # # x = self.bn2(x)
        # # x = self.relu2(x)
        # # x = self.iam_head2(x)
        # x = nn.AvgPool2d(16)(x)
        # x = x.view(x.size(0), x.size(1), -1)
        # # x = F.relu_(self.fc_cls(x))
        # # x_cls = self.cls_head(features)
        

        B, N, H, W = iam.shape
        C = features.size(1)
        
        # BxNxHxW -> BxNx(HW)
        # print(torch.max(iam.view(B, N, -1) + self.softmax_bias), torch.max((iam.view(B, N, -1) + self.softmax_bias) / self.temperature))
        iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)
        # iam_prob = F.softmax((iam.view(B, N, -1) + self.softmax_bias) / self.temperature, dim=-1)
        # if self.temperature > 1:
        #     self.temperature -= 0.5
        # iam_prob = F.softmax(iam.view(B, N, -1), dim=-1)

        # iam_prob = iam.sigmoid()
        # iam_prob = iam_prob.view(B, N, -1)
        # normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        # iam_prob = iam_prob / normalizer[:, :, None]


        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(
            iam_prob, features.view(B, C, -1).permute(0, 2, 1))

        inst_features = inst_features.reshape(
            B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
        inst_features = F.relu_(self.fc(inst_features))

        # predict classification & segmentation kernel & objectness
        # pred_logits = x_cls
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)

        # iam = iam_prob.view(B, N, H, W)

        # iam = {
        #     "iam": iam,
        # }

        return pred_logits, pred_kernel, pred_scores, iam
    

# class InstanceBranch(nn.Module):
#     def __init__(self, dim, kernel_dim, num_masks=10):
#         super().__init__()
#         dim = dim
#         num_convs = 4
#         num_masks = num_masks
#         kernel_dim = kernel_dim
        
#         self.num_groups = 1
#         self.num_classes = 2
        
#         # iam prediction, a simple conv
#         self.iam_conv = nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)
#         # # self.iam_conv = DeepIAM(dim, num_masks * self.num_groups, self.num_groups)
#         self.bn1 = nn.BatchNorm2d(num_masks)
#         self.relu1 = nn.ReLU(inplace=True)
#         # self.bn2 = nn.BatchNorm2d(num_masks)
#         # self.relu2 = nn.ReLU(inplace=True)

#         self.iam_head1 = _make_stack_3x3_convs(num_convs, num_masks, num_masks)
#         # self.iam_head2 = _make_stack_3x3_convs(num_convs, num_masks, num_masks)
        
#         expand_dim = dim * self.num_groups
#         self.fc = nn.Linear(expand_dim, expand_dim)

#         # outputs
#         # self.cls_head = ConvFCCLSHead(dim, inner_dim=256, out_dim=num_masks, num_classes=self.num_classes)
#         # self.cls_score = nn.Linear(1024, self.num_classes)
        
#         # self.fc_cls = nn.Linear(1024, 1024)

#         self.cls_score = nn.Linear(1024, self.num_classes)
#         self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
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

#         c2_xavier_fill(self.fc)

#         for m in self.iam_head1.modules():
#             if isinstance(m, nn.Conv2d):
#                 c2_msra_fill(m)

#         # for m in self.iam_head2.modules():
#         #     if isinstance(m, nn.Conv2d):
#         #         c2_msra_fill(m)


#     def forward(self, features, idx=None):
#         # predict instance activation maps
#         iam = self.iam_conv(features)

#         x = iam
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.iam_head1(x)
#         # x = nn.AvgPool2d(4)(x)
#         # x = self.bn2(x)
#         # x = self.relu2(x)
#         # x = self.iam_head2(x)
#         x = nn.AvgPool2d(16)(x)
#         x = x.view(x.size(0), x.size(1), -1)
#         # x = F.relu_(self.fc_cls(x))
#         # x_cls = self.cls_head(features)
        

#         B, N, H, W = iam.shape
#         C = features.size(1)
        
#         # BxNxHxW -> BxNx(HW)
#         # print(torch.max(iam.view(B, N, -1) + self.softmax_bias), torch.max((iam.view(B, N, -1) + self.softmax_bias) / self.temperature))
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

#         inst_features = inst_features.reshape(
#             B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
#         inst_features = F.relu_(self.fc(inst_features))

#         # predict classification & segmentation kernel & objectness
#         # pred_logits = x_cls
#         pred_logits = self.cls_score(x)
#         pred_kernel = self.mask_kernel(inst_features)
#         pred_scores = self.objectness(inst_features)

#         # iam = iam_prob.view(B, N, H, W)

#         # iam = {
#         #     "iam": iam,
#         # }

#         return pred_logits, pred_kernel, pred_scores, iam






if __name__ == "__main__":
    instance_head = InstanceBranch(dim=256, kernel_dim=10, num_masks=5)
    x = torch.rand(1, 256, 512, 512)
    # y = torch.rand(1, 5, 25, 25)
    # inst_logits, inst_kernel, inst_scores, inst_iam, occl_logits, occl_kernel, occl_scores, occl_iam = instance_head(x, y)
    # print(inst_logits.shape, occl_logits.shape)
    # print(inst_kernel.shape, occl_kernel.shape)
    # print(inst_iam.shape, occl_iam.shape)
    pred_logits, pred_kernel, pred_scores, iam = instance_head(x)
    print(pred_kernel.shape)
    print(pred_logits.shape)