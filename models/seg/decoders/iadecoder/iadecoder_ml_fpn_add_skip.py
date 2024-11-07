import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn.weight_init import c2_msra_fill
from torch.nn import init

import sys
sys.path.append("./")

from models.seg.nn.blocks import (DoubleConv, DoubleConv_v1, DoubleConv_v2, 
                                   SE_block)

from models.seg.decoders.iadecoder.iadecoder_ml_fpn import IADecoder
from configs.structure import Decoder
from utils.registry import DECODERS, HEADS
from omegaconf import OmegaConf



@DECODERS.register(name='iadecoder_ml_fpn_add_skip')
class IADecoder(IADecoder):
    def __init__(self, cfg: Decoder, embed_dims: list = [], n_levels: int = 4):
        super().__init__(cfg, embed_dims, n_levels)

        fpn_dim = 256
        self.up_conv_layers = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                in_channels = fpn_dim + 2
            else:
                in_channels = fpn_dim + 2
            out_channels = fpn_dim

            upconv = nn.Sequential(
                DoubleConv_v2(in_channels, out_channels),
                SE_block(num_features=out_channels) 
            )
            self.up_conv_layers.append(upconv)

        self._init_weights()
        

    def _forward(self, skips):
        for i in range(self.n_levels):
            if i != 0:                
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)

                skip = skips[-(i + 1)]
                skip = self.skip_conv_layers[i](skip)

                x = x + skip

                coord_features = self.compute_coordinates(x)
                x = torch.cat([coord_features, x], dim=1)

                x = self.up_conv_layers[i](x)
            else:
                skip = skips[-1]
                skip = self.skip_conv_layers[i](skip)

                coord_features = self.compute_coordinates(skip)
                x = torch.cat([coord_features, skip], dim=1)
                x = self.up_conv_layers[i](x)

            
            if i != 0:
                mask_feats = nn.UpsamplingBilinear2d(scale_factor=2)(mask_feats)
                # mask_feats = torch.cat([x, mask_feats], dim=1)
                mask_feats = mask_feats + x
                mask_feats = self.mask_branch[i](mask_feats)   
            else:
                mask_feats = self.mask_branch[i](x)


            if i != 0:
                inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
                # inst_feats = torch.cat([x, inst_feats], dim=1)
                inst_feats = inst_feats + x

                coord_features = self.compute_coordinates(inst_feats)
                inst_feats = torch.cat([coord_features, inst_feats], dim=1)
                inst_feats = self.instance_branch[i](inst_feats)
            else:
                coord_features = self.compute_coordinates(x)
                inst_feats = torch.cat([coord_features, x], dim=1)
                inst_feats = self.instance_branch[i](inst_feats)


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
    
    