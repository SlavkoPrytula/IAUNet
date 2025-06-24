import torch
import torch.nn.functional as F
from torch import nn

import math
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill
from torch.nn import init

import sys
sys.path.append('.')

from utils.tiles import unfold
from models.seg.heads.common import _make_stack_3x3_convs



class LocationHead(nn.Module):
    def __init__(self, in_channels, dim=256, kernel_dim=256, num_grids=7):
        """
        SOLOv2 Instance Head.
        """
        super().__init__()
        # fmt: off
        self.dim = dim
        self.kernel_dim = kernel_dim
        self.num_grids = num_grids
        self.num_classes = 2

        self.num_levels = 4

        self.kernel_grid_convs = _make_stack_3x3_convs(self.num_levels, in_channels+2, dim)
        self.cate_grid_convs = _make_stack_3x3_convs(self.num_levels, in_channels, dim)

        self.kernel_pred = nn.Conv2d(
            self.dim, self.kernel_dim,
            kernel_size=3, stride=1, padding=1
        )

        self.cate_pred = nn.Conv2d(
            self.dim, self.num_classes,
            kernel_size=3, stride=1, padding=1
        )

        for modules in [self.cate_pred, self.kernel_pred, self.kernel_grid_convs, self.cate_grid_convs]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)


    def forward(self, features):
        # concat coord
        x_range = torch.linspace(-1, 1, features.shape[-1], device=features.device)
        y_range = torch.linspace(-1, 1, features.shape[-2], device=features.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([features.shape[0], 1, -1, -1])
        x = x.expand([features.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        features = torch.cat([features, coord_feat], 1)

        # individual feature.
        kernel_feat = features
        seg_num_grid = self.num_grids

        kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear')
        cate_feat = kernel_feat[:, :-2, :, :]

        kernel_feat = self.kernel_grid_convs(kernel_feat)
        kernel_feat = self.kernel_pred(kernel_feat)

        cate_feat = self.cate_grid_convs(cate_feat)
        cate_feat = self.cate_pred(cate_feat)

        
        B, C, G1, G2 = kernel_feat.shape
        kernel_feat = kernel_feat.view(B, -1, C)
        cate_feat = cate_feat.view(B, -1, self.num_classes)

        return cate_feat, kernel_feat



# class AdaptiveLocationHead(nn.Module):
#     def __init__(self, in_channels, dim=256, kernel_dim=256, num_grids=7):
#         """
#         SOLOv2 Instance Head.
#         """
#         super().__init__()
#         # fmt: off
#         self.dim = dim
#         self.kernel_dim = kernel_dim
#         self.num_grids = num_grids
#         self.num_classes = 2
#         self.num_levels = 4

#         self.kernel_grid_convs = _make_stack_3x3_convs(self.num_levels, in_channels+2, dim)
#         self.cate_grid_convs = _make_stack_3x3_convs(self.num_levels, in_channels, dim)

#         self.kernel_pred = nn.Conv2d(
#             self.dim, self.kernel_dim,
#             kernel_size=3, stride=1, padding=1
#         )

#         self.cate_pred = nn.Conv2d(
#             self.dim, self.num_classes,
#             kernel_size=3, stride=1, padding=1
#         )

#         N = num_grids**2
#         self.kernel_pool = nn.ModuleList([])
#         for _ in range(N):
#             self.kernel_pool.append(
#                 nn.Sequential(
#                     nn.AdaptiveAvgPool2d(3),
#                 )
#             )

#         for modules in [self.cate_pred, self.kernel_pred, self.kernel_grid_convs, self.cate_grid_convs, self.kernel_pool]:
#             for l in modules.modules():
#                 if isinstance(l, nn.Conv2d):
#                     torch.nn.init.normal_(l.weight, std=0.01)
#                     if l.bias is not None:
#                         nn.init.constant_(l.bias, 0)


#     def forward(self, features):
#         # concat coord
#         x_range = torch.linspace(-1, 1, features.shape[-1], device=features.device)
#         y_range = torch.linspace(-1, 1, features.shape[-2], device=features.device)
#         y, x = torch.meshgrid(y_range, x_range)
#         y = y.expand([features.shape[0], 1, -1, -1])
#         x = x.expand([features.shape[0], 1, -1, -1])
#         coord_feat = torch.cat([x, y], 1)
#         features = torch.cat([features, coord_feat], 1)

#         # individual feature.
#         kernel_feat = features
#         cate_feat = kernel_feat[:, :-2, :, :]

#         kernel_feat = self.kernel_grid_convs(kernel_feat)

#         B, C, H, W = kernel_feat.shape
#         kernels = []
#         for i, b_mask in enumerate(kernel_feat):
#             b_mask = b_mask.unsqueeze(0)
#             partitioned_mask = unfold(b_mask, H//self.num_grids, W//self.num_grids)
#             partitioned_kernels = []

#             for j, p_m in enumerate(partitioned_mask):
#                 k = self.kernel_pool[j](p_m)
#                 partitioned_kernels.append(k)
#             _kernel = torch.cat(partitioned_kernels, 0)
#             kernels.append(_kernel)
#         kernels = torch.cat(kernels, dim=0)

#         kernel_feat = self.kernel_pred(kernels)


#         # print(cate_feat.shape)
#         # cate_feat = self.cate_grid_convs(cate_feat)
#         # # print(cate_feat.shape)
#         # cate_feat = self.cate_pred(cate_feat)
#         # # print(cate_feat.shape)

        
#         C = kernel_feat.shape[1]
#         kernel_feat = kernel_feat.view(B, -1, C, 3, 3)
#         # cate_feat = cate_feat.view(B, -1, self.num_classes)

#         return cate_feat, kernel_feat


class AdaptiveLocationHead(nn.Module):
    def __init__(self, in_channels, dim=256, kernel_dim=256, num_grids=7):
        """
        SOLOv2 Instance Head.
        """
        super().__init__()
        # fmt: off
        self.dim = dim
        self.kernel_dim = kernel_dim
        self.num_grids = num_grids
        self.num_classes = 2
        self.num_levels = 4

        self.kernel_grid_convs = _make_stack_3x3_convs(self.num_levels, in_channels+2, dim)
        self.cate_grid_convs = _make_stack_3x3_convs(self.num_levels, in_channels, dim)

        self.kernel_pred = nn.Conv2d(
            self.dim, self.kernel_dim*9,
            kernel_size=3, stride=1, padding=1
        )

        self.cate_pred = nn.Conv2d(
            self.dim, self.num_classes,
            kernel_size=3, stride=1, padding=1
        )

        for modules in [self.cate_pred, self.kernel_pred, self.kernel_grid_convs, self.cate_grid_convs]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)
            
            # initialize the bias for focal loss
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.cate_pred.bias, bias_value)


    def forward(self, features):
        # concat coord
        x_range = torch.linspace(-1, 1, features.shape[-1], device=features.device)
        y_range = torch.linspace(-1, 1, features.shape[-2], device=features.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([features.shape[0], 1, -1, -1])
        x = x.expand([features.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        features = torch.cat([features, coord_feat], 1)

        # individual feature.
        kernel_feat = features
        kernel_feat = nn.AdaptiveAvgPool2d(self.num_grids)(kernel_feat)
        cate_feat = kernel_feat[:, :-2, :, :]

        kernel_feat = self.kernel_grid_convs(kernel_feat)
        kernel_feat = self.kernel_pred(kernel_feat)

        cate_feat = self.cate_grid_convs(cate_feat)
        cate_feat = self.cate_pred(cate_feat)

        B, C, G1, G2 = kernel_feat.shape
        kernel_feat = kernel_feat.view(B, -1, C)
        cate_feat = cate_feat.view(B, -1, self.num_classes)

        return cate_feat, kernel_feat


class GradualLocationHead(nn.Module):
    def __init__(self, in_channels, dim=256, kernel_dim=256, num_grids=7):
        """
        SOLOv2 Instance Head.
        """
        super().__init__()
        # fmt: off
        self.dim = dim
        self.kernel_dim = kernel_dim
        # self.num_grids = num_grids
        self.num_classes = 2
        self.num_levels = 4

        self.num_grids = [256, 64, 32, 7]

        self.kernel_grid_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels+2, dim, 3, stride=1, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels+2, dim, 3, stride=1, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels+2, dim, 3, stride=1, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels+2, dim, 3, stride=1, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            ),

            # nn.Sequential(
            #     nn.Conv2d(dim, dim, 3, stride=1, padding=1),
            #     nn.GroupNorm(32, dim),
            #     nn.GELU(),
            # ),
            # nn.Sequential(
            #     nn.Conv2d(dim, dim, 3, stride=1, padding=1),
            #     nn.GroupNorm(32, dim),
            #     nn.GELU(),
            # ),
            # nn.Sequential(
            #     nn.Conv2d(dim, dim, 3, stride=1, padding=1),
            #     nn.GroupNorm(32, dim),
            #     nn.GELU(),
            # ),
        ])
        
        self.cate_grid_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, dim, 3, stride=1, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, dim, 3, stride=1, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, dim, 3, stride=1, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, dim, 3, stride=1, padding=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            ),
            # nn.Sequential(
            #     nn.Conv2d(dim, dim, 3, stride=1, padding=1),
            #     nn.GroupNorm(32, dim),
            #     nn.GELU(),
            # ),
            # nn.Sequential(
            #     nn.Conv2d(dim, dim, 3, stride=1, padding=1),
            #     nn.GroupNorm(32, dim),
            #     nn.GELU(),
            # ),
            # nn.Sequential(
            #     nn.Conv2d(dim, dim, 3, stride=1, padding=1),
            #     nn.GroupNorm(32, dim),
            #     nn.GELU(),
            # ),
        ])


        self.kernel_pred = nn.Conv2d(
            self.dim, self.kernel_dim,
            kernel_size=3, stride=1, padding=1
        )

        self.cate_pred = nn.Conv2d(
            self.dim, self.num_classes,
            kernel_size=3, stride=1, padding=1
        )

        self.fc = nn.Linear(self.kernel_dim, self.kernel_dim)
        self.kernel = nn.Linear(self.kernel_dim, self.kernel_dim)
        self.cls_score = nn.Linear(self.kernel_dim, self.num_classes)
        self.coord = nn.Linear(self.kernel_dim, 2)


        for modules in [self.cate_pred, self.kernel_pred, self.kernel_grid_convs, self.cate_grid_convs]:
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

        init.normal_(self.kernel.weight, std=0.01)
        init.constant_(self.kernel.bias, 0.0)
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

        # individual feature.
        kernel_feat = features
        # cate_feat = kernel_feat[:, :-2, :, :]

        for i in range(len(self.kernel_grid_convs)):
            kernel_feat = F.interpolate(kernel_feat, size=self.num_grids[i], mode='bilinear')

            x_range = torch.linspace(-1, 1, kernel_feat.shape[-1], device=kernel_feat.device)
            y_range = torch.linspace(-1, 1, kernel_feat.shape[-2], device=kernel_feat.device)
            y, x = torch.meshgrid(y_range, x_range)
            y = y.expand([kernel_feat.shape[0], 1, -1, -1])
            x = x.expand([kernel_feat.shape[0], 1, -1, -1])
            coord_feat = torch.cat([x, y], 1)
            kernel_feat = torch.cat([kernel_feat, coord_feat], 1)

            kernel_feat = self.kernel_grid_convs[i](kernel_feat)
        # kernel_feat = self.kernel_pred(kernel_feat)

        # for i in range(len(self.cate_grid_convs)):
        #     cate_feat = F.interpolate(cate_feat, size=self.num_grids[i], mode='bilinear')
        #     cate_feat = self.cate_grid_convs[i](cate_feat)
        # cate_feat = self.cate_pred(cate_feat)


        B, C, G1, G2 = kernel_feat.shape
        kernel_feat = kernel_feat.view(B, -1, C)

        kernel_feat = F.relu_(self.fc(kernel_feat))
        
        cate_feat = self.cls_score(kernel_feat)
        kernel_feat = self.kernel(kernel_feat)
        coord_feats = self.coord(kernel_feat)

        # cate_feat = cate_feat.view(B, -1, self.num_classes)

        return cate_feat, kernel_feat, coord_feats


if __name__ == "__main__":
    model = GradualLocationHead(256, dim=256, kernel_dim=128, num_grids=2)
    # model = AdaptiveLocationHead(32, dim=256, kernel_dim=128, num_grids=2)
    x = torch.rand(2, 256, 64, 64)
    cate, kernel, coord = model(x)
    print(cate.shape, kernel.shape, coord.shape)


