# copied from 45966022
import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn.weight_init import c2_msra_fill

import sys
sys.path.append("./")

from models.seg.decoders.iadecoder_v2.iadecoder import IADecoder as BaseDecoder
from configs.structure import Decoder
from utils.registry import DECODERS, HEADS


@DECODERS.register(name="iadecoder_ml_v2")
class IADecoder(BaseDecoder):
    def __init__(self, cfg: Decoder, embed_dims: list = [], n_levels: int = 4):
        super().__init__(cfg, embed_dims, n_levels)

        # instance head.
        self.instance_head = nn.ModuleList([])
        for i in range(self.n_levels):
            instance_head = HEADS.build(cfg.instance_head)
            self.instance_head.append(instance_head)
        
        self._init_weights()


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

            if i != 0:
                results = self.instance_head[i](inst_feats, mask_feats, inst_embed)
                inst_embed = results["kernels"]['instance_kernel']
            else:
                results = self.instance_head[i](inst_feats, mask_feats)
                inst_embed = results["kernels"]['instance_kernel']

            mask_feats = results['pixel_feats']

                    
        mask_feats = self.projection(mask_feats)
        results["mask_feats"] = mask_feats
        results["inst_feats"] = inst_feats
    
        return results
