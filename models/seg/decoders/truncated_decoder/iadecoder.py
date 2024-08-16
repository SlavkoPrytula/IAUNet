import torch
from torch import nn

import sys
sys.path.append("./")

from ...nn.blocks import (DoubleConv_v1)
from ...heads.instance_head import InstanceBranch
from ...heads.mask_head import MaskBranch

from models.seg.decoders.base import BaseDecoder
from configs import cfg
from utils.registry import DECODERS, HEADS


@DECODERS.register(name='truncated_decoder-iadecoder')
class IADecoder(BaseDecoder):
    def __init__(self, cfg: cfg, embed_dims: list = [], n_levels: int = 4):
        super().__init__(cfg, embed_dims, n_levels)
        
        self.bridge = nn.Sequential(
            DoubleConv_v1(embed_dims[-1], embed_dims[-2]),
        )
        
        embed_dims = embed_dims[::-1]
        self.up_conv_layers = nn.ModuleList([])
        for i in range(self.n_levels-1):
            in_channels = embed_dims[i+1] * 2

            if i != self.n_levels - 2:
                out_channels = embed_dims[i+2]
            else: 
                out_channels = embed_dims[i+1]
                
            upconv = nn.Sequential(
                DoubleConv_v1(in_channels, out_channels)
            )
            self.up_conv_layers.append(upconv)

        embed_dims = embed_dims[1:] + [embed_dims[-1]]

        # mask branch.
        self.mask_branch = nn.ModuleList([])
        for i in range(self.n_levels-1):
            if i == 0:
                mask_branch = MaskBranch(
                    in_channels=embed_dims[i+1], 
                    out_channels=self.mask_dim, 
                    num_convs=self.num_convs
                    )
            else:
                mask_branch = MaskBranch(
                    in_channels=embed_dims[i+1] + self.mask_dim, 
                    out_channels=self.mask_dim, 
                    num_convs=self.num_convs
                    )
            self.mask_branch.append(mask_branch)
        self.projection = nn.Conv2d(self.mask_dim, self.kernel_dim, kernel_size=1)
        
        # instance branch.
        self.instance_branch = nn.ModuleList([])
        for i in range(self.n_levels-1):
            if i == 0:
                instance_branch = InstanceBranch(
                    in_channels=embed_dims[i+1] + 2, 
                    out_channels=self.mask_dim, 
                    num_convs=self.num_convs
                    )
            else:
                instance_branch = InstanceBranch(
                    in_channels=embed_dims[i+1] + self.mask_dim + 2, 
                    out_channels=self.mask_dim, 
                    num_convs=self.num_convs
                    )
            self.instance_branch.append(instance_branch)

        # instance head.
        self.instance_head = HEADS.build(cfg.instance_head)

        self._init_weights()


    def forward(self, skips, ori_shape):
        results, mask_feats = self._forward(skips, ori_shape)
        results = self.process_outputs(results, mask_feats, ori_shape)

        return results
    

    def _forward(self, skips, ori_shape):
        x = self.bridge(skips[-1])
        skips = skips[:-1]

        # go up
        for i in range(self.n_levels-1):
            x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
            x = torch.cat([x, skips[-(i + 1)]], dim=1)
            x = self.up_conv_layers[i](x)

            
            if i != 0:
                mask_feats = nn.UpsamplingBilinear2d(scale_factor=2)(mask_feats)
                mask_feats = torch.cat([x, mask_feats], dim=1)
                mask_feats = self.mask_branch[i](mask_feats)
            else:
                mask_feats = self.mask_branch[i](x)

            if i != 0:
                inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
                inst_feats = torch.cat([x, inst_feats], dim=1)
                coord_features = self.compute_coordinates(inst_feats)

                inst_feats = torch.cat([coord_features, inst_feats], dim=1)
                inst_feats = self.instance_branch[i](inst_feats)
            else:
                coord_features = self.compute_coordinates(x)
                inst_feats = torch.cat([coord_features, x], dim=1)
                inst_feats = self.instance_branch[i](inst_feats)
        
        # out layer.
        results = self.instance_head(inst_feats)
        mask_feats = self.projection(mask_feats)

        return results, mask_feats
