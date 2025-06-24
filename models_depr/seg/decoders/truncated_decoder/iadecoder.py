import torch
from torch import nn
from torch.nn import init
from fvcore.nn.weight_init import c2_msra_fill

import sys
sys.path.append("./")

from ...nn.blocks import (DoubleConv_v1)
from ...heads.instance_head import InstanceBranch
from ...heads.mask_head import MaskBranch

from models.seg.decoders.base import BaseDecoder
from configs.structure import Decoder
from utils.registry import DECODERS, HEADS


@DECODERS.register(name='truncated_decoder-iadecoder')
class IADecoder(BaseDecoder):
    def __init__(self, cfg: Decoder, embed_dims: list = [], n_levels: int = 4):
        super().__init__(cfg, embed_dims, n_levels)
        
        print(embed_dims)
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
        mask_branch_layer = HEADS.get(cfg.mask_branch.type)
        self.mask_branch = nn.ModuleList([])
        for i in range(self.n_levels-1):
            if i == 0:
                mask_branch = mask_branch_layer(
                    in_channels=embed_dims[i+1], 
                    out_channels=self.mask_dim, 
                    num_convs=self.num_convs
                    )
            else:
                mask_branch = mask_branch_layer(
                    in_channels=embed_dims[i+1] + self.mask_dim, 
                    out_channels=self.mask_dim, 
                    num_convs=self.num_convs
                    )
            self.mask_branch.append(mask_branch)
        self.projection = nn.Conv2d(self.mask_dim, self.kernel_dim, kernel_size=1)
        
        # instance branch.
        instance_branch_layer = HEADS.get(cfg.instance_branch.type)
        self.instance_branch = nn.ModuleList([])
        for i in range(self.n_levels-1):
            if i == 0:
                instance_branch = instance_branch_layer(
                    in_channels=embed_dims[i+1] + 2, 
                    out_channels=self.mask_dim, 
                    num_convs=self.num_convs
                    )
            else:
                instance_branch = instance_branch_layer(
                    in_channels=embed_dims[i+1] + self.mask_dim + 2, 
                    out_channels=self.mask_dim, 
                    num_convs=self.num_convs
                    )
            self.instance_branch.append(instance_branch)

        # instance head.
        self.instance_head = HEADS.build(cfg.instance_head)

        self._init_weights()


    def _init_weights(self):
        for modules in [self.up_conv_layers, self.bridge]:
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


    def forward(self, skips, ori_shape):
        results = self._forward(skips, ori_shape)
        results = self.process_outputs(results, ori_shape)

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

        results["mask_feats"] = mask_feats
        results["inst_feats"] = inst_feats

        return results
