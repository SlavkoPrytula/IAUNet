# copied from 45966022
import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn.weight_init import c2_msra_fill

import sys
sys.path.append("./")

from models.seg.decoders.base import BaseDecoder
from configs.structure import Decoder
from utils.registry import DECODERS, HEADS

from ...nn.blocks import DoubleConv_v2, SE_block


@DECODERS.register(name="iadecoder_v2")
class IADecoder(BaseDecoder):
    def __init__(self, cfg: Decoder, embed_dims: list = [], n_levels: int = 4):
        super().__init__(cfg, embed_dims, n_levels)
        
        self.num_convs = cfg.num_convs
        self.coord_conv = cfg.coord_conv
        self.embed_dims = embed_dims
        self.n_levels = n_levels
        self.last_layer_only = cfg.last_layer_only

        self.mask_dim = cfg.mask_branch.dim
        self.inst_dim = cfg.instance_branch.dim
        self.kernel_dim = cfg.instance_head.kernel_dim

        assert self.inst_dim == self.mask_dim, "mask dim should be equal to instance dim!"

        self.bridge = nn.Sequential(
            nn.Conv2d(embed_dims[-1], embed_dims[-1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[-1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            
        )

        embed_dims = self.embed_dims[::-1]

        self.up_conv_layers = nn.ModuleList([])
        for i in range(self.n_levels):
            in_channels = embed_dims[i] * 2 + 2
            out_channels = embed_dims[i+1]

            upconv = nn.Sequential(
                DoubleConv_v2(in_channels, out_channels),
                SE_block(num_features=out_channels)
            )
            self.up_conv_layers.append(upconv)

        embed_dims = embed_dims[1:] + [embed_dims[-1]]

        # mask branch.
        mask_branch_layer = HEADS.get(cfg.mask_branch.type)
        self.mask_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                mask_branch = mask_branch_layer(
                    in_channels=embed_dims[i], 
                    out_channels=self.mask_dim, 
                    num_convs=self.num_convs
                )
            else:
                mask_branch = mask_branch_layer(
                    in_channels=embed_dims[i] + self.mask_dim, 
                    out_channels=self.mask_dim, 
                    num_convs=self.num_convs
                )
            self.mask_branch.append(mask_branch)

        self.projection = nn.Conv2d(self.mask_dim, self.kernel_dim, kernel_size=1)
        
        # instance branch.
        self.instance_branch = nn.ModuleList([])
        instance_branch_layer = HEADS.get(cfg.instance_branch.type)
        for i in range(self.n_levels):
            if i == 0:
                instance_branch = instance_branch_layer(
                    in_channels=embed_dims[i], 
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

        # instance branch.
        self.instance_head = HEADS.build(cfg.instance_head)

        self._init_weights()


    def _init_weights(self):
        for modules in [self.up_conv_layers, self.bridge]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)
        c2_msra_fill(self.projection)

        
    def forward(self, skips, ori_shape):
        results = self._forward(skips, ori_shape)
        results = self.process_outputs(results, ori_shape)

        return results


    def _forward(self, skips, ori_shape):
        x = skips[-1]
        x = self.bridge(x)

        for i in range(self.n_levels):
            coord_features = self.compute_coordinates(x)
            x = torch.cat([coord_features, x], dim=1)
            
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
                inst_feats = self.instance_branch[i](x)
                    
        
        results = self.instance_head(inst_feats)
        mask_feats = self.projection(mask_feats)
        results["mask_feats"] = mask_feats
        results["inst_feats"] = inst_feats
    
        return results
