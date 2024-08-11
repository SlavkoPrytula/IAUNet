import torch
from torch import nn

import sys
sys.path.append("./")

from models.seg.decoders.base import BaseDecoder
from configs.structure import Decoder
from utils.registry import HEADS, DECODERS


@DECODERS.register(name='iadecoder_ml')
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
        x = skips[-1]

        # go up
        for i in range(self.n_levels):
            if i != 0:
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
            coord_features = self.compute_coordinates(x)
            x = torch.cat([coord_features, x], dim=1)

            x = torch.cat([x, skips[-(i + 1)]], dim=1)
            x = self.up_conv_layers[i](x)

            
            # multi-level
            # if self.multi_level:
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

            if i != 0:
                results = self.instance_head[i](inst_feats, inst_embed)
                inst_embed = results["kernels"]['instance_kernel']
            else:
                results = self.instance_head[i](inst_feats)
                inst_embed = results["kernels"]['instance_kernel']

         # out layer.
        mask_feats = self.projection(mask_feats)

        results = self.process_outputs(results, mask_feats, ori_shape)
        
        return results


        