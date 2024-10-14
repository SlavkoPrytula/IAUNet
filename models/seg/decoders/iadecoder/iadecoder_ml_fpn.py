import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn.weight_init import c2_msra_fill
from torch.nn import init

import sys
sys.path.append("./")

from models.seg.nn.blocks import (DoubleConv, DoubleConv_v1, DoubleConv_v2, 
                                   SE_block)

from models.seg.decoders.iadecoder.iadecoder import IADecoder
from configs.structure import Decoder
from utils.registry import DECODERS, HEADS


@DECODERS.register(name='iadecoder_ml_fpn')
class IADecoder(IADecoder):
    def __init__(self, cfg: Decoder, embed_dims: list = [], n_levels: int = 4):
        super().__init__(cfg, embed_dims, n_levels)

        self.coord_conv = cfg.coord_conv
        self.num_convs = cfg.num_convs

        self.mask_dim = cfg.mask_branch.dim
        self.inst_dim = cfg.instance_branch.dim
        self.kernel_dim = cfg.instance_head.kernel_dim
        self.cfg = cfg  

        self.embed_dims = embed_dims
        self.skips = True

        embed_dims = self.embed_dims[::-1]
        # out_dims = [256, 256, 256, 256]

        self.skip_conv_layers = nn.ModuleList([])
        for i in range(self.n_levels):
            skip_in_channels = embed_dims[i]
            skip_out_channels = 256

            skip_conv = nn.Conv2d(skip_in_channels, skip_out_channels, kernel_size=1)
            self.skip_conv_layers.append(skip_conv)

        
        self.up_conv_layers = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                in_channels = 256 + 2
            else:
                in_channels = 256 + 256 + 2
            out_channels = 256

            upconv = nn.Sequential(
                DoubleConv_v1(in_channels, out_channels),
                SE_block(num_features=out_channels)
            )
            self.up_conv_layers.append(upconv)


        embed_dims = embed_dims[1:] + [embed_dims[-1]]

        # mask branch.
        mask_dim = cfg.mask_branch.dim
        mask_branch_layer = HEADS.get(cfg.mask_branch.type)
        self.mask_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                mask_branch = mask_branch_layer(
                    in_channels=256, 
                    out_channels=self.mask_dim, 
                    num_convs=self.num_convs
                )
            else:
                mask_branch = mask_branch_layer(
                    in_channels=256 + self.mask_dim, 
                    out_channels=self.mask_dim, 
                    num_convs=self.num_convs
                )
            self.mask_branch.append(mask_branch)

        self.projection = nn.Conv2d(mask_dim, self.kernel_dim, kernel_size=1)
        
        # instance features.
        self.instance_branch = nn.ModuleList([])
        instance_branch_layer = HEADS.get(cfg.instance_branch.type)
        for i in range(self.n_levels):
            if i == 0:
                instance_branch = instance_branch_layer(
                    in_channels=256, 
                    out_channels=self.inst_dim, 
                    num_convs=self.num_convs
                )
            else:
                instance_branch = instance_branch_layer(
                    in_channels=256 + self.inst_dim + 2, 
                    out_channels=self.inst_dim, 
                    num_convs=self.num_convs
                )
            self.instance_branch.append(instance_branch)

        # instance head.
        self.instance_head = nn.ModuleList([])
        for i in range(self.n_levels):
            instance_head = HEADS.build(cfg.instance_head)
            self.instance_head.append(instance_head)

        self._init_weights()


    def _init_weights(self):
        for modules in [self.up_conv_layers]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)
        c2_msra_fill(self.projection)


    def _forward(self, skips):
        for i in range(self.n_levels):
            if i != 0:
                coord_features = self.compute_coordinates(x)
                x = torch.cat([coord_features, x], dim=1)
                
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)

                skip = skips[-(i + 1)]
                skip = self.skip_conv_layers[i](skip)

                x = torch.cat([x, skip], dim=1)
                x = self.up_conv_layers[i](x)
            else:
                skip = skips[-1]
                skip = self.skip_conv_layers[i](skip)

                coord_features = self.compute_coordinates(skip)
                x = torch.cat([coord_features, skip], dim=1)
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


            if i != 0:
                results = self.instance_head[i](inst_feats, mask_feats, inst_embed)
                inst_embed = results["inst_feats"]['instance_feats']
            else:
                results = self.instance_head[i](inst_feats, mask_feats)
                inst_embed = results["inst_feats"]['instance_feats']

            mask_feats = results['mask_pixel_feats']
            inst_feats = results['inst_pixel_feats']

    
        mask_feats = self.projection(mask_feats)
        results["mask_feats"] = mask_feats
        results["inst_feats"] = inst_feats
    
        return results
    