import torch
from torch import nn

import sys
sys.path.append("./")

from models.seg.decoders.base import BaseDecoder
from configs.structure import Decoder
from utils.registry import HEADS, DECODERS


# here we only use the green instance branch for decoding instances
# this reasonably enables us to controll the instance embedings and correcponding instance features 
# 
@DECODERS.register(name='iadecoder_green')
class IADecoder(BaseDecoder):
    def __init__(self, cfg: Decoder, embed_dims: list = [], n_levels: int = 4):
        super().__init__(cfg, embed_dims, n_levels)
       
        # instance head.
        self.instance_head = nn.ModuleList([])
        for i in range(self.n_levels):
            instance_head = HEADS.build(cfg.instance_head)
            self.instance_head.append(instance_head)

        self._init_weights()


    def forward(self, skips, ori_shape):
        results = self._forward(skips, ori_shape)
        results = self.process_outputs(results, ori_shape)

        return results
    

    def _forward(self, skips, ori_shape):
        x = skips[-1]

        for i in range(self.n_levels):
            if i != 0:
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)

            x = torch.cat([x, skips[-(i + 1)]], dim=1)
            x = self.up_conv_layers[i](x)

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

            if i != 0:
                results = self.instance_head[i](inst_feats, inst_embed)
                inst_embed = results["kernels"]['instance_kernel']
            else:
                results = self.instance_head[i](inst_feats)
                inst_embed = results["kernels"]['instance_kernel']
            
            inst_feats = results['pixel_feats']
        
        # results = self.instance_head(inst_feats)
        # mask_feats = results['pixel_feats']
        results["mask_feats"] = inst_feats

        return results
