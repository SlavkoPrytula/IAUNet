import torch
from torch import nn

import sys
sys.path.append("./")

from models.seg.decoders.iadecoder_v2.iadecoder import IADecoder
from configs.structure import Decoder
from utils.registry import HEADS, DECODERS


@DECODERS.register(name='iadecoder_ml_v2')
class IADecoder(IADecoder):
    def __init__(self, cfg: Decoder, embed_dims: list = [], n_levels: int = 4):
        super().__init__(cfg, embed_dims, n_levels)

        # instance head.
        self.instance_head = nn.ModuleList([])
        for i in range(self.n_levels):
            # cfg.instance_head.in_channels = embed_dims[i]
            instance_head = HEADS.build(cfg.instance_head)
            self.instance_head.append(instance_head)

        self._init_weights()
    

    def _forward(self, skips):
        for i in range(self.n_levels):
            if i != 0:    
                skip = skips[-(i + 1)]
                skip = self.skip_conv_layers[i](skip)

                coord_features = self.compute_coordinates(x)
                x = torch.cat([coord_features, x], dim=1)
                
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
                # x = self.up_layers[i](x)

                x = torch.cat([x, skip], dim=1)
                x = self.up_conv_layers[i](x)
            else:
                skip = skips[-1]
                skip = self.skip_conv_layers[i](skip)

                coord_features = self.compute_coordinates(skip)
                x = torch.cat([coord_features, skip], dim=1)
                # x = self.up_layers[i](x)

                x = self.up_conv_layers[i](x)


            if i != 0:
                results = self.instance_head[i](x, inst_embed)
                inst_embed = results["inst_feats"]['instance_feats']
            else:
                results = self.instance_head[i](x)
                inst_embed = results["inst_feats"]['instance_feats']

            x = results['pixel_feats']

        mask_feats = self.projection(x)
        results["mask_feats"] = mask_feats
        results["inst_feats"] = x
    
        return results
    