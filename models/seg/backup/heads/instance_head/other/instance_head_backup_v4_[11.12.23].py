import torch 
from torch import nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F

import math
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

import sys
sys.path.append('.')

# from models.seg.blocks import ContextBlock
from models.seg.heads.common import _make_stack_3x3_convs
from models.seg.heads.common import Block
from models.seg.heads.instance_head.iam import IAM, DeepIAM

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
    
# NOTE: swapped with best version below
# class InstanceBranch(nn.Module):
#     def __init__(self, dim, kernel_dim, num_masks=10, out_dim=1024):
#         super().__init__()
#         dim = dim
#         num_masks = num_masks
#         kernel_dim = kernel_dim
        
#         self.num_groups = 1
#         self.num_classes = 2
        
#         # iam prediction, a group conv
#         expand_dim = dim * self.num_groups
        
#         self.iam_conv = nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)
#         self.bn = nn.BatchNorm2d(num_masks)
#         self.relu = nn.ReLU(inplace=True)


#         # self.iam_convs = nn.ModuleList([
#         #     nn.Sequential(
#         #         nn.Conv2d(dim, num_masks, kernel_size=3, padding=1),
#         #         nn.BatchNorm2d(num_masks),
#         #         nn.ReLU(inplace=True)
#         #     ),
#         #     nn.Sequential(
#         #         nn.Conv2d(num_masks, num_masks, kernel_size=3, padding=1),
#         #         nn.BatchNorm2d(num_masks),
#         #         nn.ReLU(inplace=True)
#         #     ),
#         #     nn.Sequential(
#         #         nn.Conv2d(num_masks, num_masks, kernel_size=3, padding=1),
#         #     ),
#         # ])


#         # self.iam_conv = nn.Conv2d(dim, num_masks * self.num_groups, 1, groups=self.num_groups)
#         # self.iam_conv = IAM(dim, num_masks)
#         # self.iam_conv = nn.Sequential(
#         #     Block(dim),
#         #     nn.Conv2d(dim, num_masks, 3, padding=1),
#         #     # nn.Conv2d(num_masks, num_masks, 3, padding=1, groups=num_masks),
#         #     # nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups, bias=True)
#         #     # nn.Conv2d(num_masks * self.num_groups, num_masks * self.num_groups, 1, groups=self.num_groups)
#         # )
        
#         # outputs
#         self.fc = nn.Linear(expand_dim, expand_dim)

#         self.cls_score = nn.Linear(out_dim, self.num_classes)
#         self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
#         self.objectness = nn.Linear(out_dim, 1)
#         self.coords = nn.Linear(out_dim, 2)

#         self.prior_prob = 0.01
#         self._init_weights()

#         # for modules in [self.iam_convs]:
#         #     for l in modules.modules():
#         #         if isinstance(l, nn.Conv2d):
#         #             init.normal_(l.weight, std=0.01)
#         #             if l.bias is not None:
#         #                 nn.init.constant_(l.bias, 0)


#     def _init_weights(self):
#         bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
#         for module in [self.iam_conv, self.cls_score]:
#             init.constant_(module.bias, bias_value)
#         init.normal_(self.iam_conv.weight, std=0.1)

#         # for module in [self.cls_score]:
#         #     init.constant_(module.bias, bias_value)

#         init.normal_(self.cls_score.weight, std=0.01)
#         init.normal_(self.mask_kernel.weight, std=0.01)
#         init.constant_(self.mask_kernel.bias, 0.0)
#         c2_xavier_fill(self.fc)
        

#     def forward(self, features, iam=None, return_iam_only=False, reduction=16):
#         # predict instance activation maps
#         # if iam is not None:
#         #     _iam = self.iam_conv(features)
#         #     iam = _iam + iam

#         # if return_iam_only:
#         #     iam = self.iam_conv(features)
#         #     return iam

#         iam = self.iam_conv(features)
        
#         # iam = self.iam_convs[0](features)
        
#         x = iam
#         x = self.bn(x)
#         x = self.relu(x)
#         x = nn.AvgPool2d(reduction)(x)
#         x = x.view(x.size(0), x.size(1), -1)

#         # iam = self.iam_convs[1](iam)
#         # iam = self.iam_convs[2](iam)

#         iam_prob = iam.sigmoid()

#         B, N, H, W = iam.shape
#         C = features.size(1)

#         # iam_prob = F.softmax(iam.view(B, N, -1), dim=-1)
        
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
        
#         inst_features = F.relu_(self.fc(inst_features))  # potentially add point-wise convs

#         # predict classification & segmentation kernel & objectness
#         pred_logits = self.cls_score(x)
#         pred_kernel = self.mask_kernel(inst_features)
#         pred_scores = self.objectness(x)
#         pred_coords = self.coords(x)

#         # iam = iam_prob.view(B, N, H, W)
#         # iam = {
#         #     "iam": iam,
#         # }

#         return pred_logits, pred_kernel, pred_scores, iam, pred_coords

#         # (N, H, W)  (C, H, W)
#         # (N, H, W)


class ConvFCCLSHead(nn.Module):
    def __init__(self, dim, inner_dim=256, out_dim=256, num_classes=1, num_levels=4):
        """
        SOLOv2 Instance Head.
        """
        super().__init__()
        # fmt: off
        self.dim = dim
        self.inner_dim = inner_dim
        self.num_classes = num_classes
        self.num_levels = num_levels

        self.cate_grid_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, inner_dim, 3, stride=1, padding=1),
                nn.BatchNorm2d(inner_dim),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(inner_dim, inner_dim, 3, stride=1, padding=1),
                nn.BatchNorm2d(inner_dim),
                nn.ReLU(inplace=True),
            ),
            # nn.Sequential(
            #     nn.Conv2d(inner_dim, inner_dim, 3, stride=1, padding=1),
            #     nn.BatchNorm2d(inner_dim),
            #     nn.ReLU(inplace=True),
            # ),
            # nn.Sequential(
            #     nn.Conv2d(inner_dim, inner_dim, 3, stride=1, padding=1),
            #     nn.BatchNorm2d(inner_dim),
            #     nn.ReLU(inplace=True),
            # ),
        ])

        self.cate_pred = nn.Conv2d(
            inner_dim, out_dim, 
            kernel_size=3, stride=1, padding=1
        )

        self.fc = nn.Linear(1024, out_dim)
        self.cls_score = nn.Linear(out_dim, self.num_classes)


        for modules in [self.cate_pred, self.cate_grid_convs]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)
            
        # initialize the bias for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        init.constant_(self.cate_pred.bias, bias_value)

        for module in [self.cls_score]:
            init.constant_(module.bias, bias_value)

        c2_xavier_fill(self.fc)


    def forward(self, features):
        # concat coord
        # x_range = torch.linspace(-1, 1, features.shape[-1], device=features.device)
        # y_range = torch.linspace(-1, 1, features.shape[-2], device=features.device)
        # y, x = torch.meshgrid(y_range, x_range)
        # y = y.expand([features.shape[0], 1, -1, -1])
        # x = x.expand([features.shape[0], 1, -1, -1])
        # coord_feat = torch.cat([x, y], 1)
        # features = torch.cat([features, coord_feat], 1)
        
        cate_feat = features

        for i in range(len(self.cate_grid_convs)):  
            # cate_feat = F.interpolate(cate_feat, size=self.num_grids[i], mode='bilinear')
            cate_feat = nn.AvgPool2d(4)(cate_feat)
            cate_feat = self.cate_grid_convs[i](cate_feat)
        cate_feat = self.cate_pred(cate_feat)

        B, C, H, W = cate_feat.shape
        cate_feat = cate_feat.view(B, C, -1)

        cate_feat = F.relu_(self.fc(cate_feat))
        cate_feat = self.cls_score(cate_feat)

        return cate_feat


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

        self.iam_head1 = _make_stack_3x3_convs(num_convs, num_masks, num_masks)
        # self.iam_head2 = _make_stack_3x3_convs(num_convs, num_masks, num_masks)
        
        expand_dim = dim * self.num_groups
        self.fc = nn.Linear(expand_dim, expand_dim)

        # outputs
        self.cls_head = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, num_masks, 3, padding=1),
            nn.BatchNorm2d(num_masks),
            nn.ReLU(inplace=True),
        )
        # self.cls_head = _make_stack_3x3_convs(2, dim, dim)
        # self.cls_head = ConvFCCLSHead(dim, inner_dim=256, out_dim=num_masks, num_classes=self.num_classes)
        # self.cls_score = nn.Linear(1024, self.num_classes)
        
        self.fc_cls = nn.Linear(1024, 1024)

        # self.cls_score = nn.Linear(expand_dim, self.num_classes)
        self.cls_score = nn.Linear(1024, self.num_classes)
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

        for m in self.iam_head1.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

        # for m in self.iam_head2.modules():
        #     if isinstance(m, nn.Conv2d):
        #         c2_msra_fill(m)


    def forward(self, features, idx=None):
        # predict instance activation maps
        iam = self.iam_conv(features)

        # x_cls = self.cls_head(features)
        # x_cls = nn.AvgPool2d(16)(x_cls)
        # x_cls = x_cls.view(x_cls.size(0), x_cls.size(1), -1)
        
        # x_cls = self.cls_score(x_cls)
        # print(x_cls.shape)
        x = iam
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.iam_head1(x)
        # x = nn.AvgPool2d(4)(x)
        # x = self.bn2(x)
        # x = self.relu2(x)
        # x = self.iam_head2(x)
        x = nn.AvgPool2d(16)(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = F.relu_(self.fc_cls(x))
        # x_cls = self.cls_head(features)
        

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
        # pred_logits = self.cls_score(inst_features)
        # pred_logits = self.cls_score(x_cls)
        pred_logits = self.cls_score(x)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)

        # iam = iam_prob.view(B, N, H, W)

        # iam = {
        #     "iam": iam,
        # }

        return pred_logits, pred_kernel, pred_scores, iam



class InstanceBranchNoIAM(nn.Module):
    def __init__(self, dim, kernel_dim):
        super().__init__()
        dim = dim
        kernel_dim = kernel_dim
        
        self.num_groups = 1
        self.num_classes = 2
        
        expand_dim = dim * self.num_groups
        self.fc = nn.Linear(expand_dim, expand_dim)
        self.cls_score = nn.Linear(expand_dim, self.num_classes)
        self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
        self.objectness = nn.Linear(expand_dim, 1)
        self.coords = nn.Linear(expand_dim, 2)

        self.prior_prob = 0.01
        self._init_weights()


    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.cls_score]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.cls_score.weight, std=0.01)
        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)
        c2_xavier_fill(self.fc)
        

    def forward(self, features, iam):
        # iam_prob = iam.sigmoid()

        B, N, H, W = iam.shape
        C = features.size(1)

        iam_prob = F.softmax(iam.view(B, N, -1), dim=-1)
        
        # BxNxHxW -> BxNx(HW)
        # iam_prob = iam_prob.view(B, N, -1)
        # normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        # iam_prob = iam_prob / normalizer[:, :, None]

        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(
            iam_prob, features.view(B, C, -1).permute(0, 2, 1))

        # inst_features = inst_features.reshape(
        #     B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(
        #     B, N // self.num_groups, -1)
        
        inst_features = F.relu_(self.fc(inst_features))  # potentially add point-wise convs

        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(x)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        pred_coords = self.coords(inst_features)

        return pred_logits, pred_kernel, pred_scores, iam, pred_coords



class Attention(nn.Module):
    def __init__(self, in_dim, out_channels):
        super(Attention, self).__init__()
        self.chanel_in = in_dim
        self.out_channels = out_channels
        
        self.key_conv = nn.Conv2d(in_channels=self.chanel_in, out_channels=self.out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=self.chanel_in, out_channels=self.out_channels, kernel_size=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.maxpool(x)
        x = self.maxpool(x)
        x = self.maxpool(x)
        B, C, W, H = x.size()

        key = self.key_conv(x).view(-1, 1, H*W)      # (BN, 1, HW)
        value = self.value_conv(x).view(-1, H*W, 1)  # (BN, HW, 1)

        attn = torch.matmul(value, key)
        # attn = self.softmax(energy) # (HW, HW)

        attn = attn.mean(1).view(B, -1, H, W)
        attn = F.interpolate(attn, size=[H*8, W*8], mode='bilinear', align_corners=False)

        return attn



class CentextAttention(nn.Module):
    def __init__(self, in_channels, out_channels=5):
        super(CentextAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_mask = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
    
    def forward(self, x):
        context_mask = self.conv_mask(x)
        
        return context_mask


# from models.seg.blocks.gcn import ContextBlock
from models.seg.heads.common import kaiming_init, last_zero_init
class ContextBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 ratio
                 ):
        super(ContextBlock, self).__init__()
        
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)

        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        self.channel_mul_conv = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_mask, mode='fan_in')
        self.conv_mask.inited = True
        last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        out = x
        context = self.spatial_pool(x)
        # channel_mul_term = self.channel_mul_conv(context)
        channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
        out = out * channel_mul_term

        return out



# NOTE: [main]
# class InstanceBranch(nn.Module):
#     def __init__(self, dim, kernel_dim, num_masks=10):
#         super().__init__()
#         dim = dim
#         num_masks = num_masks
#         kernel_dim = kernel_dim
        
#         self.num_groups = 1
#         self.num_classes = 1
        
#         # iam prediction, a simple conv
#         # self.iam_conv = nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)
#         self.iam_conv = nn.Sequential(
#             # Block(dim),
#             # CentextAttention(dim, num_masks)
#             # Attention(dim, num_masks)
#             # nn.Conv2d(dim, num_masks * self.num_groups, 1, padding=1, groups=self.num_groups)
#             # ContextBlock(inplanes=dim, ratio=4),
#             nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)
#             # nn.Conv2d(dim, num_masks * self.num_groups, 1, groups=self.num_groups)

#             # nn.Conv2d(dim, num_masks, 3, padding=1),
#             # nn.BatchNorm2d(num_masks),
#             # nn.ReLU(inplace=True),
#             # nn.Conv2d(num_masks, num_masks, 3, padding=1),
#         )
#         # self.iam_conv = DeepIAM(dim, num_masks * self.num_groups, self.num_groups)
#         self.inst_conv = nn.Conv2d(dim, dim, 3, padding=1)
        
#         expand_dim = dim * self.num_groups
#         self.fc = nn.Linear(expand_dim, expand_dim)

#         # outputs
#         self.cls_score = nn.Linear(expand_dim, self.num_classes)
#         self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
#         self.objectness = nn.Linear(expand_dim, 1)
        
#         self.prior_prob = 0.01
#         self._init_weights()
        
#         self.softmax_bias = nn.Parameter(torch.ones([1, ]), requires_grad=True)
#         # self.temprature = nn.Parameter(torch.ones([1, ]) * 30, requires_grad=True)
#         self.temprature = 30


#     def _init_weights(self):
#         bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
#         for module in [self.cls_score]:
#             init.constant_(module.bias, bias_value)

#         for modules in [self.iam_conv, self.inst_conv]:
#             for l in modules.modules():
#                 if isinstance(l, nn.Conv2d):
#                     nn.init.normal_(l.weight, std=0.01)
#                     if l.bias is not None:
#                         nn.init.constant_(l.bias, bias_value)

#         init.normal_(self.cls_score.weight, std=0.01)
#         init.normal_(self.mask_kernel.weight, std=0.01)
#         init.constant_(self.mask_kernel.bias, 0.0)
#         c2_xavier_fill(self.fc)


#     def forward(self, features, idx=None):
#         # predict instance activation maps
#         iam = self.iam_conv(features)
#         # features = self.inst_conv(features)

#         B, N, H, W = iam.shape
#         C = features.size(1)
        
#         # BxNxHxW -> BxNx(HW)
#         # iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)
#         # iam_prob = F.softmax(iam.view(B, N, -1) / self.temprature, dim=-1)
#         iam_prob = F.softmax(iam.view(B, N, -1), dim=-1)


#         # aggregate features: BxCxHxW -> Bx(HW)xC
#         inst_features = torch.bmm(
#             iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        
#         # nxhw * hwxc -> n*c
#         inst_features = inst_features.reshape(
#             B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
#         inst_features = F.relu_(self.fc(inst_features))

#         # predict classification & segmentation kernel & objectness
#         pred_logits = self.cls_score(inst_features)
#         pred_kernel = self.mask_kernel(inst_features)
#         pred_scores = self.objectness(inst_features)

#         # iam = iam_prob.view(B, N, H, W)

#         # iam = {
#         #     "iam": iam,
#         # }

#         return pred_logits, pred_kernel, pred_scores, iam




# ==========================================================================
from utils.tiles import fold, unfold

# class InstanceBranch(nn.Module):
#     def __init__(self, dim, kernel_dim, num_masks=10):
#         super().__init__()
#         dim = dim
#         kernel_dim = kernel_dim
#         self.num_masks = num_masks
        
#         self.num_groups = 1
#         self.num_classes = 1
        
#         # iam prediction, a simple conv
#         # self.iam_conv = nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)
#         self.iam_conv = nn.Sequential(
#             nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)
#         )
        
#         expand_dim = dim * self.num_groups
#         self.fc = nn.Linear(expand_dim, expand_dim)

#         # outputs
#         self.cls_score = nn.Linear(expand_dim, self.num_classes)
#         self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
#         self.objectness = nn.Linear(expand_dim, 1)
        
#         self.prior_prob = 0.01
#         self._init_weights()
        
#         self.softmax_bias = nn.Parameter(torch.ones([1, ]), requires_grad=True)


#     def _init_weights(self):
#         bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
#         for module in [self.cls_score]:
#             init.constant_(module.bias, bias_value)

#         for modules in [self.iam_conv]:
#             for l in modules.modules():
#                 if isinstance(l, nn.Conv2d):
#                     nn.init.normal_(l.weight, std=0.01)
#                     if l.bias is not None:
#                         nn.init.constant_(l.bias, bias_value)

#         init.normal_(self.cls_score.weight, std=0.01)
#         init.normal_(self.mask_kernel.weight, std=0.01)
#         init.constant_(self.mask_kernel.bias, 0.0)
#         c2_xavier_fill(self.fc)


#     def forward(self, features, idx=None):
#         # (B, C, H, W) -> (N, B, C, H//4, W//4)
#         kernel_size = 128
#         stride = 32
#         # patch the input feature map.
#         features = unfold(features, kernel_size=kernel_size, stride=stride)
#         N, B, C, H, W = features.shape
#         features = features.contiguous()

#         # local-scope activation maps
#         iams = torch.zeros(N, B, self.num_masks, H, W, device=features.device)
#         for i in range(features.shape[0]):
#             iams[i] = self.iam_conv(features[i])

#         # fold back the overlapping patches
#         features = features.view(N, B*C, H, W)
#         features = fold(features, out_size=512, kernel_size=kernel_size, stride=stride)
#         features = features.view(B, C, features.shape[2], features.shape[3])
        
#         iams = iams.view(N, B*self.num_masks, H, W)
#         iams = fold(iams, out_size=512, kernel_size=kernel_size, stride=stride)
#         iams = iams.view(B, self.num_masks, iams.shape[2], iams.shape[3])
            

#         B, N, H, W = iams.shape
#         C = features.size(1)
        
#         # BxNxHxW -> BxNx(HW)
#         iam_prob = F.softmax(iams.view(B, N, -1), dim=-1)

#         # aggregate features: BxCxHxW -> Bx(HW)xC
#         inst_features = torch.bmm(
#             iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        
#         # nxhw * hwxc -> n*c
#         inst_features = inst_features.reshape(
#             B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
#         inst_features = F.relu_(self.fc(inst_features))

#         # predict classification & segmentation kernel & objectness
#         pred_logits = self.cls_score(inst_features)
#         pred_kernel = self.mask_kernel(inst_features)
#         pred_scores = self.objectness(inst_features)

#         return pred_logits, pred_kernel, pred_scores, iams



# class InstanceBranch(nn.Module):
#     def __init__(self, dim, kernel_dim, num_masks=10):
#         super().__init__()
#         dim = dim
#         num_masks = num_masks
#         kernel_dim = kernel_dim
        
#         self.num_groups = 1
#         self.num_classes = 1
        
#         # iam prediction, a simple conv
#         self.iam_conv = nn.Conv2d(dim, num_masks, 3, padding=1)
#         self.inst_conv = nn.Conv2d(dim, num_masks, 3, padding=1)

#         self.pool = nn.AdaptiveAvgPool2d((3, 3))
#         self.mask_kernel = nn.Conv2d(num_masks, num_masks, 3, padding=1)
        
#         expand_dim = dim * self.num_groups
#         self.fc = nn.Linear(expand_dim, expand_dim)

#         # outputs
#         self.cls_score = nn.Linear(expand_dim, self.num_classes)
#         # self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
#         self.objectness = nn.Linear(expand_dim, 1)
        
#         self.prior_prob = 0.01
#         self._init_weights()
        
#         self.softmax_bias = nn.Parameter(torch.ones([1, ]), requires_grad=True)
#         self.temprature = 30

#     # def _init_weights(self):
#     #     bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
#     #     for module in [self.iam_conv, self.cls_score]:
#     #         init.constant_(module.bias, bias_value)
#     #     init.normal_(self.iam_conv.weight, std=0.01)
#     #     init.normal_(self.cls_score.weight, std=0.01)
#     #     init.normal_(self.mask_kernel.weight, std=0.01)
#     #     init.constant_(self.mask_kernel.bias, 0.0)

#     def _init_weights(self):
#         bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
#         for module in [self.iam_conv, self.cls_score]:
#             init.constant_(module.bias, bias_value)
#         init.normal_(self.iam_conv.weight, std=0.01)
#         init.normal_(self.cls_score.weight, std=0.01)

#         init.normal_(self.mask_kernel.weight, std=0.01)
#         init.constant_(self.mask_kernel.bias, 0.0)
#         c2_xavier_fill(self.fc)

#     # def _init_weights(self):
#     #     bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
#     #     for module in [self.cls_score]:
#     #         init.constant_(module.bias, bias_value)

#     #     for modules in [self.iam_conv, self.inst_conv]:
#     #         for l in modules.modules():
#     #             if isinstance(l, nn.Conv2d):
#     #                 nn.init.normal_(l.weight, std=0.01)
#     #                 if l.bias is not None:
#     #                     nn.init.constant_(l.bias, bias_value)

#     #     init.normal_(self.cls_score.weight, std=0.01)

#     #     init.normal_(self.mask_kernel.weight, std=0.01)
#     #     init.constant_(self.mask_kernel.bias, 0.0)
#     #     c2_xavier_fill(self.fc)


#     def forward(self, features, idx=None):
#         # predict instance activation maps
#         iam = self.iam_conv(features)
#         features = self.inst_conv(features)

#         B, N, H, W = iam.shape
#         C = features.size(1)
        
#         # BxNxHxW -> BxNx(HW)
#         iam_prob = F.softmax(iam.view(B, N, -1), dim=-1)

#         # aggregate features: BxCxHxW -> Bx(HW)xC
#         inst_features = torch.matmul(
#             iam_prob.view(B, C, H, W), 
#             features #.view(B, C, -1).permute(0, 2, 1)
#         ) # (B, N, H, W)
#         print(inst_features.shape)

#         kernel_feats = self.mask_kernel(features) # (B, N, H, W)
#         pred_kernel = self.pool(kernel_feats) # (B, N, 3, 3)
#         print(pred_kernel.shape)

#         print(inst_features.shape)
#         inst_features = inst_features.view(B, N, -1) # (B, N, HW)
#         print(inst_features.shape)
#         inst_features = F.relu_(self.fc(inst_features))

#         # predict classification & segmentation kernel & objectness
#         pred_logits = self.cls_score(inst_features)
#         # pred_kernel = self.mask_kernel(inst_features)
#         pred_scores = self.objectness(inst_features)

#         iam = {
#             "iam": iam,
#         }

#         return pred_logits, pred_kernel, pred_scores, iam

# ================================================================




# # class GroupInstanceBranch(nn.Module):
# #     def __init__(self, kernel_dim, num_masks=10):
# #         super().__init__()
# #         dim = 256
# #         num_convs = 4
# #         num_masks = num_masks
# #         kernel_dim = kernel_dim
        
# #         self.num_groups = 2
# #         self.num_classes = 1
        
# #         # iam prediction, a simple conv
# #         self.iam_conv = nn.Conv2d(dim, num_masks, 3, padding=1)
# #         self.iam_conv_1 = nn.Conv2d(dim, num_masks, 3, padding=1)
        
# #         expand_dim = 2 * dim * self.num_groups
# #         self.fc = nn.Linear(expand_dim, expand_dim)

# #         # outputs
# #         self.cls_score = nn.Linear(expand_dim, self.num_classes)
# #         self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
# #         self.objectness = nn.Linear(expand_dim, 1)
        
# #         self.prior_prob = 0.01
# #         self._init_weights()
        
# #         self.softmax_bias = nn.Parameter(torch.ones([1, ]))

# #     def _init_weights(self):
# #         bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
# #         for module in [self.iam_conv, self.iam_conv_1, self.cls_score]:
# #             init.constant_(module.bias, bias_value)
# #         init.normal_(self.iam_conv.weight, std=0.01)
# #         init.normal_(self.iam_conv_1.weight, std=0.01)
# #         init.normal_(self.cls_score.weight, std=0.01)
# #         init.normal_(self.mask_kernel.weight, std=0.01)
# #         init.constant_(self.mask_kernel.bias, 0.0)


# #     def forward(self, inst_feats, ovlp_feats, idx=None):
# #         # 
# #         # Xi --> IAM_0 --\
# #         #                + --> IAM
# #         # Xo --> IAM_1 --/
# #         # 
        
# #         # predict instance activation maps
# #         inst_iam = self.iam_conv(inst_feats)
# #         ovlp_iam = self.iam_conv_1(ovlp_feats)
# #         iam = torch.cat([inst_iam, ovlp_iam], dim=1)  # [1, 50, 512, 512]
# #         # iam = inst_iam + ovlp_iam

# #         inst_feats = torch.cat([inst_feats, ovlp_feats], dim=1)  # [1, 512, 512, 512] 

# #         B, N, H, W = iam.shape
# #         C = inst_feats.size(1)
        
# #         # BxNxHxW -> BxNx(HW)
# #         iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)

# #         # aggregate features: BxCxHxW -> Bx(HW)xC
# #         inst_features = torch.bmm(
# #             iam_prob, inst_feats.view(B, C, -1).permute(0, 2, 1))

# #         # (2N)(HW) x (HW)xC -> (2N)x(2C)
# #         # (2, N, (2C)) -> (N, 2, (2C)) -> (N, (4C))

# #         inst_features = inst_features.reshape(
# #             B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
# #         inst_features = F.relu_(self.fc(inst_features))

# #         # predict classification & segmentation kernel & objectness
# #         pred_logits = self.cls_score(inst_features)
# #         pred_kernel = self.mask_kernel(inst_features)
# #         pred_scores = self.objectness(inst_features)

# #         # iam = iam_prob.view(B, N, H, W)

# #         iam = {
# #             "iam": iam,
# #             "instance_iam": inst_iam,
# #             "overlap_iam": ovlp_iam
# #         }

# #         return pred_logits, pred_kernel, pred_scores, iam



# # class GroupInstanceBranch(nn.Module):
# #     def __init__(self, kernel_dim, num_masks=10):
# #         super().__init__()
# #         dim = 256
# #         num_convs = 4
# #         num_masks = num_masks
# #         kernel_dim = kernel_dim
        
# #         self.num_groups = 1
# #         self.num_classes = 1
        
# #         # iam prediction, a simple conv
# #         self.iam_conv = nn.Conv2d(dim + num_masks, num_masks, 3, padding=1)
# #         self.iam_conv_1 = nn.Conv2d(dim, num_masks, 3, padding=1)
        
# #         expand_dim = dim * self.num_groups + num_masks
# #         self.fc = nn.Linear(expand_dim, expand_dim)

# #         # outputs
# #         self.cls_score = nn.Linear(expand_dim, self.num_classes)
# #         self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
# #         self.objectness = nn.Linear(expand_dim, 1)
        
# #         self.prior_prob = 0.01
# #         self._init_weights()
        
# #         self.softmax_bias = nn.Parameter(torch.ones([1, ]))

# #     def _init_weights(self):
# #         bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
# #         for module in [self.iam_conv, self.iam_conv_1, self.cls_score]:
# #             init.constant_(module.bias, bias_value)
# #         init.normal_(self.iam_conv.weight, std=0.01)
# #         init.normal_(self.iam_conv_1.weight, std=0.01)
# #         init.normal_(self.cls_score.weight, std=0.01)
# #         init.normal_(self.mask_kernel.weight, std=0.01)
# #         init.constant_(self.mask_kernel.bias, 0.0)


# #     def forward(self, inst_feats, ovlp_feats, idx=None):
# #         ovlp_feats = inst_feats.clone()

# #         # predict instance activation maps
# #         # ovlp.
# #         ovlp_iam = self.iam_conv_1(ovlp_feats)
# #         inst_feats = torch.cat([inst_feats, ovlp_iam], dim=1)  # [1, 356, 512, 512] 
        
# #         # inst.
# #         iam = self.iam_conv(inst_feats)

# #         # iam = torch.cat([inst_iam, ovlp_iam], dim=1)  # [1, 50, 512, 512]
# #         # iam = inst_iam + ovlp_iam
# #         # inst_feats = torch.cat([inst_feats, ovlp_feats], dim=1)  # [1, 512, 512, 512] 

# #         B, N, H, W = iam.shape
# #         C = inst_feats.size(1)
        
# #         # BxNxHxW -> BxNx(HW)
# #         iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)

# #         # aggregate features: BxCxHxW -> Bx(HW)xC
# #         inst_features = torch.bmm(
# #             iam_prob, inst_feats.view(B, C, -1).permute(0, 2, 1))
        
# #         # (N, HW) x (HW, D) -> (N, D)
# #         # fc:  (N, D) -> (N, D)
# #         # cls: (N, D) -> (N, 1)
# #         # ker: (N, D) -> (N, C)
# #         # obj: (N, D) -> (N, 1)

# #         inst_features = inst_features.reshape(
# #             B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
# #         inst_features = F.relu_(self.fc(inst_features))

# #         # predict classification & segmentation kernel & objectness
# #         pred_logits = self.cls_score(inst_features)
# #         pred_kernel = self.mask_kernel(inst_features)
# #         pred_scores = self.objectness(inst_features)

# #         # iam = iam_prob.view(B, N, H, W)

# #         iam = {
# #             # "iam": iam,
# #             # "instance_iam": iam,
# #             # "overlap_iam": ovlp_iam
# #         }

# #         return pred_logits, pred_kernel, pred_scores, iam


# # NOTE: base version
class GroupInstanceBranch(nn.Module):
    def __init__(self, dim, kernel_dim, num_masks=10):
        super().__init__()
        dim = dim
        num_masks = num_masks
        kernel_dim = kernel_dim
        
        self.num_groups = 1
        self.num_classes = 2
        
        # iam prediction, a simple conv
        self.iam_conv = nn.Conv2d(dim, num_masks, 3, padding=1)
        
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


    def forward(self, inst_feats, idx=None):
        # predict instance activation maps
        iam = self.iam_conv(inst_feats)

        B, N, H, W = iam.shape
        C = inst_feats.size(1)
        
        # BxNxHxW -> BxNx(HW)
        iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)

        # iam_prob = iam.sigmoid()        

        # iam_prob = iam_prob.view(B, N, -1)
        # normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        # iam_prob = iam_prob / normalizer[:, :, None]

        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(
            iam_prob, inst_feats.view(B, C, -1).permute(0, 2, 1))
        
        # (N, HW) x (HW, D) -> (N, D)
        # fc:  (N, D) -> (N, D)
        # cls: (N, D) -> (N, 1)
        # ker: (N, D) -> (N, C)
        # obj: (N, D) -> (N, 1)

        inst_features = inst_features.reshape(
            B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
        inst_features = F.relu_(self.fc(inst_features))

        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)

        # iam = iam_prob.view(B, N, H, W)

        return pred_logits, pred_kernel, pred_scores, iam




# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()

#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)
#         return self.sigmoid(x)
    

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
           
#         self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
#                                nn.ReLU(),
#                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)
    

# # class GroupInstanceBranch(nn.Module):
# #     def __init__(self, dim, kernel_dim, num_masks=10):
# #         super().__init__()
# #         dim = dim
# #         num_masks = num_masks
# #         kernel_dim = kernel_dim
        
# #         self.num_groups = 1
# #         self.num_classes = 1
        
# #         # iam prediction, a simple conv
# #         # self.iam_context_block = ContextBlock(
# #         #     inplanes=256, 
# #         #     ratio=4, 
# #         #     pooling_type='att', 
# #         #     fusion_types=['channel_add', 'channel_mul']
# #         #     )
# #         self.ca = ChannelAttention(256)
# #         self.sa = SpatialAttention()
# #         self.iam_conv = nn.Conv2d(dim, num_masks, 3, padding=1)
        
# #         expand_dim = dim * self.num_groups
# #         self.fc = nn.Linear(expand_dim, expand_dim)

# #         # outputs
# #         self.cls_score = nn.Linear(expand_dim, self.num_classes)
# #         self.mask_kernel = nn.Linear(expand_dim, kernel_dim)
# #         self.objectness = nn.Linear(expand_dim, 1)
        
# #         self.prior_prob = 0.01
# #         self._init_weights()
        
# #         self.softmax_bias = nn.Parameter(torch.ones([1, ]))

# #     def _init_weights(self):
# #         bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
# #         for module in [self.iam_conv, self.cls_score]:
# #             init.constant_(module.bias, bias_value)
# #         init.normal_(self.iam_conv.weight, std=0.01)
# #         init.normal_(self.cls_score.weight, std=0.01)
# #         init.normal_(self.mask_kernel.weight, std=0.01)
# #         init.constant_(self.mask_kernel.bias, 0.0)


# #     def forward(self, inst_feats, idx=None):
# #         # predict instance activation maps
# #         inst_feats = self.ca(inst_feats) * inst_feats
# #         inst_feats = self.sa(inst_feats) * inst_feats
# #         iam = self.iam_conv(inst_feats)

# #         B, N, H, W = iam.shape
# #         C = inst_feats.size(1)
        
# #         # BxNxHxW -> BxNx(HW)
# #         iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)

# #         # aggregate features: BxCxHxW -> Bx(HW)xC
# #         inst_features = torch.bmm(
# #             iam_prob, inst_feats.view(B, C, -1).permute(0, 2, 1))
        
# #         # (N, HW) x (HW, D) -> (N, D)
# #         # fc:  (N, D) -> (N, D)
# #         # cls: (N, D) -> (N, 1)
# #         # ker: (N, D) -> (N, C)
# #         # obj: (N, D) -> (N, 1)

# #         inst_features = inst_features.reshape(
# #             B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
# #         inst_features = F.relu_(self.fc(inst_features))

# #         # predict classification & segmentation kernel & objectness
# #         pred_logits = self.cls_score(inst_features)
# #         pred_kernel = self.mask_kernel(inst_features)
# #         pred_scores = self.objectness(inst_features)

# #         # iam = iam_prob.view(B, N, H, W)

# #         return pred_logits, pred_kernel, pred_scores, iam




class InstanceIAMFusionBranch(nn.Module):
    def __init__(self, dim, kernel_dim, num_masks=10):
        super().__init__()
        dim = dim
        num_masks = num_masks
        kernel_dim = kernel_dim
        
        self.num_groups = 1
        self.num_classes = 1
        
        # iam prediction, a simple conv
        self.inst_iam_conv = nn.ModuleList([
            nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups),
            nn.Conv2d(num_masks * self.num_groups, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)
        ])

        self.occl_iam_conv = nn.ModuleList([
            nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups),
            nn.Conv2d(num_masks * self.num_groups, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)
        ])
        
        expand_dim = dim * self.num_groups
        self.inst_fc = nn.Linear(expand_dim, expand_dim)
        self.occl_fc = nn.Linear(expand_dim, expand_dim)

        # outputs
        self.inst_cls_score = nn.Linear(expand_dim, self.num_classes)
        self.inst_mask_kernel = nn.Linear(expand_dim, kernel_dim)
        self.inst_objectness = nn.Linear(expand_dim, 1)

        self.occl_cls_score = nn.Linear(expand_dim, self.num_classes)
        self.occl_mask_kernel = nn.Linear(expand_dim, kernel_dim)
        self.occl_objectness = nn.Linear(expand_dim, 1)

        
        self.prior_prob = 0.01
        
        self.inst_softmax_bias = nn.Parameter(torch.ones([1, ]))
        self.occl_softmax_bias = nn.Parameter(torch.ones([1, ]))

        self._init_weights()

        
    def _init_weights(self):
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)

        for module in self.inst_iam_conv + self.occl_iam_conv:
            # if isinstance(m, nn.Conv2d):
            init.constant_(module.bias, bias_value)
            init.normal_(module.weight, std=0.01)

        for module in [self.inst_cls_score, self.occl_cls_score]:
            init.constant_(module.bias, bias_value)
        
        init.normal_(self.inst_cls_score.weight, std=0.01)
        init.normal_(self.occl_cls_score.weight, std=0.01)

        init.normal_(self.inst_mask_kernel.weight, std=0.01)
        init.constant_(self.inst_mask_kernel.bias, 0.0)
        init.normal_(self.occl_mask_kernel.weight, std=0.01)
        init.constant_(self.occl_mask_kernel.bias, 0.0)
        
        c2_xavier_fill(self.inst_fc)
        c2_xavier_fill(self.occl_fc)


    def forward(self, inst_feats, occl_feats, idx=None):
        # predict instance activation maps

        inst_iam = self.inst_iam_conv[0](inst_feats)
        occl_iam = self.occl_iam_conv[0](occl_feats)

        _inst_iam = inst_iam.clone()
        _occl_iam = occl_iam.clone()

        occl_iam = occl_iam + _inst_iam
        inst_iam = inst_iam + _occl_iam

        inst_iam = self.inst_iam_conv[1](inst_iam)
        occl_iam = self.occl_iam_conv[1](occl_iam)

        B, N, H, W = inst_iam.shape
        C = inst_feats.size(1)
        
        # BxNxHxW -> BxNx(HW)
        inst_iam_prob = F.softmax(inst_iam.view(B, N, -1) + self.inst_softmax_bias, dim=-1)
        occl_iam_prob = F.softmax(occl_iam.view(B, N, -1) + self.occl_softmax_bias, dim=-1)


        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(
            inst_iam_prob, inst_feats.view(B, C, -1).permute(0, 2, 1))
        
        occl_features = torch.bmm(
            occl_iam_prob, occl_feats.view(B, C, -1).permute(0, 2, 1))


        inst_features = inst_features.reshape(
            B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
        occl_features = occl_features.reshape(
            B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
        inst_features = F.relu_(self.inst_fc(inst_features))
        occl_features = F.relu_(self.occl_fc(occl_features))


        # predict classification & segmentation kernel & objectness
        pred_inst_logits = self.inst_cls_score(inst_features)
        pred_inst_kernel = self.inst_mask_kernel(inst_features)
        pred_inst_scores = self.inst_objectness(inst_features)

        pred_occl_logits = self.occl_cls_score(occl_features)
        pred_occl_kernel = self.occl_mask_kernel(occl_features)
        pred_occl_scores = self.occl_objectness(occl_features)

        # iam = iam_prob.view(B, N, H, W)

        return pred_inst_logits, pred_inst_kernel, pred_inst_scores, inst_iam, pred_occl_logits, pred_occl_kernel, pred_occl_scores, occl_iam
    


# class DoubleIAMBranch(nn.Module):
#     def __init__(self, dim, kernel_dim, num_masks=10):
#         super().__init__()
#         dim = dim
#         num_masks = num_masks
#         kernel_dim = kernel_dim
        
#         self.num_groups = 1
#         self.num_classes = 1
        
#         # iam prediction, a simple conv
#         self.iam_conv = nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)
        
#         expand_dim = dim * self.num_groups
#         self.inst_fc = nn.Linear(expand_dim, expand_dim)
#         self.occl_fc = nn.Linear(expand_dim, expand_dim)

#         # outputs
#         self.inst_cls_score = nn.Linear(expand_dim, self.num_classes)
#         self.inst_mask_kernel = nn.Linear(expand_dim, kernel_dim)
#         self.inst_objectness = nn.Linear(expand_dim, 1)

#         self.occl_cls_score = nn.Linear(expand_dim, self.num_classes)
#         self.occl_mask_kernel = nn.Linear(expand_dim, kernel_dim)
#         self.occl_objectness = nn.Linear(expand_dim, 1)

        
#         self.prior_prob = 0.01
        
#         self.inst_softmax_bias = nn.Parameter(torch.ones([1, ]))
#         self.occl_softmax_bias = nn.Parameter(torch.ones([1, ]))

#         self._init_weights()

        
#     def _init_weights(self):
#         bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)

#         for module in [self.iam_conv]:
#             # if isinstance(m, nn.Conv2d):
#             init.constant_(module.bias, bias_value)
#             init.normal_(module.weight, std=0.01)

#         for module in [self.inst_cls_score, self.occl_cls_score]:
#             init.constant_(module.bias, bias_value)
        
#         init.normal_(self.inst_cls_score.weight, std=0.01)
#         init.normal_(self.occl_cls_score.weight, std=0.01)

#         init.normal_(self.inst_mask_kernel.weight, std=0.01)
#         init.constant_(self.inst_mask_kernel.bias, 0.0)
#         init.normal_(self.occl_mask_kernel.weight, std=0.01)
#         init.constant_(self.occl_mask_kernel.bias, 0.0)
        
#         c2_xavier_fill(self.inst_fc)
#         c2_xavier_fill(self.occl_fc)


#     def forward(self, inst_feats, occl_feats, idx=None):
#         # predict instance activation maps

#         occl_iam = self.iam_conv(occl_feats)
#         inst_iam = self.iam_conv(inst_feats)

#         B, N, H, W = inst_iam.shape
#         C = inst_feats.size(1)
        
#         # BxNxHxW -> BxNx(HW)
#         inst_iam_prob = F.softmax(inst_iam.view(B, N, -1) + self.inst_softmax_bias, dim=-1)
#         occl_iam_prob = F.softmax(occl_iam.view(B, N, -1) + self.occl_softmax_bias, dim=-1)


#         # aggregate features: BxCxHxW -> Bx(HW)xC
#         inst_features = torch.bmm(
#             inst_iam_prob, inst_feats.view(B, C, -1).permute(0, 2, 1))
        
#         occl_features = torch.bmm(
#             occl_iam_prob, occl_feats.view(B, C, -1).permute(0, 2, 1))


#         inst_features = inst_features.reshape(
#             B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
#         occl_features = occl_features.reshape(
#             B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
#         inst_features = F.relu_(self.inst_fc(inst_features))
#         occl_features = F.relu_(self.occl_fc(occl_features))


#         # predict classification & segmentation kernel & objectness
#         pred_inst_logits = self.inst_cls_score(inst_features)
#         pred_inst_kernel = self.inst_mask_kernel(inst_features)
#         pred_inst_scores = self.inst_objectness(inst_features)

#         pred_occl_logits = self.occl_cls_score(occl_features)
#         pred_occl_kernel = self.occl_mask_kernel(occl_features)
#         pred_occl_scores = self.occl_objectness(occl_features)

#         # iam = iam_prob.view(B, N, H, W)

#         return pred_inst_logits, pred_inst_kernel, pred_inst_scores, inst_iam, pred_occl_logits, pred_occl_kernel, pred_occl_scores, occl_iam
    


class DoubleIAMBranch(nn.Module):
    def __init__(self, dim, kernel_dim, num_masks=10):
        super().__init__()
        dim = dim
        num_masks = num_masks
        kernel_dim = kernel_dim
        
        self.num_groups = 1
        self.num_classes = 2
        
        # iam prediction, a simple conv
        self.iam_conv = nn.Conv2d(dim, num_masks * self.num_groups, 3, padding=1, groups=self.num_groups)
        
        expand_dim = dim * self.num_groups
        self.inst_fc = nn.Linear(expand_dim, expand_dim)
        self.occl_fc = nn.Linear(expand_dim, expand_dim)

        # outputs
        self.inst_cls_score = nn.Linear(expand_dim, self.num_classes)
        self.inst_mask_kernel = nn.Linear(expand_dim, kernel_dim)
        self.inst_objectness = nn.Linear(expand_dim, 1)

        self.occl_cls_score = nn.Linear(expand_dim, self.num_classes)
        self.occl_mask_kernel = nn.Linear(expand_dim, kernel_dim)
        self.occl_objectness = nn.Linear(expand_dim, 1)

        
        self.prior_prob = 0.01
        
        self.inst_softmax_bias = nn.Parameter(torch.ones([1, ]))
        self.occl_softmax_bias = nn.Parameter(torch.ones([1, ]))

        self._init_weights()

        
    def _init_weights(self):
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)

        for module in [self.iam_conv]:
            # if isinstance(m, nn.Conv2d):
            init.constant_(module.bias, bias_value)
            init.normal_(module.weight, std=0.01)

        for module in [self.inst_cls_score, self.occl_cls_score]:
            init.constant_(module.bias, bias_value)
        
        init.normal_(self.inst_cls_score.weight, std=0.01)
        init.normal_(self.occl_cls_score.weight, std=0.01)

        init.normal_(self.inst_mask_kernel.weight, std=0.01)
        init.constant_(self.inst_mask_kernel.bias, 0.0)
        init.normal_(self.occl_mask_kernel.weight, std=0.01)
        init.constant_(self.occl_mask_kernel.bias, 0.0)
        
        c2_xavier_fill(self.inst_fc)
        c2_xavier_fill(self.occl_fc)


    def forward(self, inst_feats, occl_feats, idx=None):
        # predict instance activation maps
        occl_iam = self.iam_conv(occl_feats)
        inst_iam = self.iam_conv(inst_feats)

        B, N, H, W = inst_iam.shape
        C = inst_feats.size(1)
        
        inst_iam_prob = inst_iam.sigmoid()
        
        # BxNxHxW -> BxNx(HW)
        inst_iam_prob = inst_iam_prob.view(B, N, -1)
        normalizer = inst_iam_prob.sum(-1).clamp(min=1e-6)
        inst_iam_prob = inst_iam_prob / normalizer[:, :, None]


        occl_iam_prob = occl_iam.sigmoid()
        
        # BxNxHxW -> BxNx(HW)
        occl_iam_prob = occl_iam_prob.view(B, N, -1)
        normalizer = occl_iam_prob.sum(-1).clamp(min=1e-6)
        occl_iam_prob = occl_iam_prob / normalizer[:, :, None]

        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(
            inst_iam_prob, inst_feats.view(B, C, -1).permute(0, 2, 1))
        
        occl_features = torch.bmm(
            occl_iam_prob, occl_feats.view(B, C, -1).permute(0, 2, 1))


        inst_features = inst_features.reshape(
            B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
        occl_features = occl_features.reshape(
            B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
        inst_features = F.relu_(self.inst_fc(inst_features))
        occl_features = F.relu_(self.occl_fc(occl_features))


        # predict classification & segmentation kernel & objectness
        pred_inst_logits = self.inst_cls_score(inst_features)
        pred_inst_kernel = self.inst_mask_kernel(inst_features)
        pred_inst_scores = self.inst_objectness(inst_features)

        pred_occl_logits = self.occl_cls_score(occl_features)
        pred_occl_kernel = self.occl_mask_kernel(occl_features)
        pred_occl_scores = self.occl_objectness(occl_features)

        # iam = iam_prob.view(B, N, H, W)

        return pred_inst_logits, pred_inst_kernel, pred_inst_scores, inst_iam, pred_occl_logits, pred_occl_kernel, pred_occl_scores, occl_iam
    




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



