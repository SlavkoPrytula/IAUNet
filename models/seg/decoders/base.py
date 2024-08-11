import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn.weight_init import c2_msra_fill
from torch.nn import init

from abc import ABC, abstractmethod

import sys
sys.path.append("./")

from ..heads.instance_head import InstanceBranch
from ..heads.mask_head import MaskBranch

from ..nn.blocks import (DoubleConv, DoubleConv_v1)

from configs.structure import Decoder
from utils.registry import HEADS


class BaseDecoder(nn.Module, ABC):
    def __init__(self, cfg: Decoder, embed_dims: list = [], n_levels: int = 4):
        super(BaseDecoder, self).__init__()
        self.num_convs = cfg.num_convs
        self.coord_conv = cfg.coord_conv
        self.embed_dims = embed_dims
        self.n_levels = n_levels
        self.last_layer_only = cfg.last_layer_only

        self.mask_dim = cfg.mask_branch.dim
        self.inst_dim = cfg.instance_branch.dim
        self.kernel_dim = cfg.instance_head.kernel_dim

        assert self.inst_dim == self.mask_dim, "mask dim should be equal to instance dim!"

        embed_dims = embed_dims[::-1]

        self.up_conv_layers = nn.ModuleList([])
        for i in range(self.n_levels):
            in_channels = embed_dims[i] * 2 + 2

            if i != self.n_levels - 1:
                out_channels = embed_dims[i+1]
            else:
                out_channels = embed_dims[i]

            upconv = DoubleConv_v1(in_channels, out_channels)
            self.up_conv_layers.append(upconv)

        embed_dims = embed_dims[1:] + [embed_dims[-1]]

        # mask branch.
        mask_branch_layer = HEADS.get(cfg.mask_branch.type)
        self.mask_branch = nn.ModuleList([])
        if not self.last_layer_only:
            for i in range(self.n_levels):
                if i == 0:
                    mask_branch = mask_branch_layer(
                        embed_dims[i], 
                        out_channels=self.mask_dim, 
                        num_convs=self.num_convs
                        )
                else:
                    mask_branch = mask_branch_layer(
                        embed_dims[i] + self.mask_dim, 
                        out_channels=self.mask_dim, 
                        num_convs=self.num_convs
                        )
                self.mask_branch.append(mask_branch)
        else:
            mask_branch = mask_branch_layer(
                embed_dims[-1],
                out_channels=self.mask_dim, 
                num_convs=self.num_convs
                )
            self.mask_branch.append(mask_branch)

        self.projection = nn.Conv2d(self.mask_dim, self.kernel_dim, kernel_size=1)
        
        # instance branch.
        instance_branch_layer = HEADS.get(cfg.instance_branch.type)
        self.instance_branch = nn.ModuleList([])
        if not self.last_layer_only:
            for i in range(self.n_levels):
                if i == 0:
                    instance_branch = instance_branch_layer(
                        in_channels=embed_dims[i] + 2, 
                        out_channels=self.inst_dim, 
                        num_convs=self.num_convs
                        )
                else:
                    instance_branch = instance_branch_layer(
                        in_channels=embed_dims[i] + self.inst_dim + 2, 
                        out_channels=self.inst_dim, 
                        num_convs=self.num_convs
                        )
                self.instance_branch.append(instance_branch)
        else:
            instance_branch = instance_branch_layer(
                in_channels=embed_dims[-1] + 2, 
                out_channels=self.inst_dim, 
                num_convs=self.num_convs
                )
            self.instance_branch.append(instance_branch)

        # instance head.
        self.instance_head = HEADS.build(cfg.instance_head)


    def _init_weights(self):
        for modules in [self.up_conv_layers]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)

        c2_msra_fill(self.projection)

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        x_loc = torch.linspace(-1, 1, h, device=x.device)
        y_loc = torch.linspace(-1, 1, w, device=x.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x_loc, y_loc], 1)

        return coord_feat

    @abstractmethod
    def forward(skips, ori_shape):
        ...

    def process_outputs(self, results, mask_feats, ori_shape):
        logits = results["logits"]
        scores = results["objectness_scores"]
        inst_kernel = results["kernels"]["instance_kernel"]
        bboxes = results["bboxes"]['instance_bboxes']
        
        # instance masks.
        N = inst_kernel.shape[1]
        B, C, H, W = mask_feats.shape

        inst_masks = torch.bmm(inst_kernel, mask_feats.view(B, C, H * W))
        inst_masks = inst_masks.view(B, N, H, W)
        bboxes = bboxes.sigmoid()

        inst_masks = F.interpolate(inst_masks, size=ori_shape[-2:], 
                                   mode="bilinear", align_corners=False)

        output = {
            'pred_logits': logits,
            'pred_scores': scores,
            'pred_iams': results['iams'],
            'pred_instance_masks': inst_masks,
            'pred_bboxes': bboxes,
        }
    
        return output