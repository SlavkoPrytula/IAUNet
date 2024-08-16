import torch
from torch import nn

import sys
sys.path.append("./")

from models.seg.decoders.base import BaseDecoder
from configs.structure import Decoder
from utils.registry import DECODERS
import time


@DECODERS.register(name='iadecoder_timed')
class IADecoder(BaseDecoder):
    def __init__(self, cfg: Decoder, embed_dims: list = [], n_levels: int = 4):
        super().__init__(cfg, embed_dims, n_levels)
        self._init_weights()

    def forward(self, skips, ori_shape):
        start_time = time.time()  # Start timing the full forward pass
        results, mask_feats = self._forward(skips, ori_shape)
        results = self.process_outputs(results, mask_feats, ori_shape)
        total_time = time.time() - start_time  # End timing the full forward pass

        print(f"Total Forward Pass Time: {total_time:.6f} seconds")
        return results
    
    def _forward(self, skips, ori_shape):
        times = {
            'skips_concat': [], 
            'mask_feats_concat': [], 
            'inst_feats_concat': [],
            'coord': [], 
            'mask_branch': [], 
            'inst_branch': [],
            'up_conv_layers': [],
            'instance_head': []
        }

        x = skips[-1]

        for i in range(self.n_levels):
            # Timing skips_concat
            start = time.time()
            if i != 0:
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
            x = torch.cat([x, skips[-(i + 1)]], dim=1)
            times['skips_concat'].append(time.time() - start)

            # Timing convolution
            start = time.time()
            x = self.up_conv_layers[i](x)
            times['up_conv_layers'].append(time.time() - start)

            if not self.last_layer_only:
                # Timing mask_feats_concat
                if i != 0:
                    start = time.time()
                    mask_feats = nn.UpsamplingBilinear2d(scale_factor=2)(mask_feats)
                    mask_feats = torch.cat([x, mask_feats], dim=1)
                    times['mask_feats_concat'].append(time.time() - start)

                    start = time.time()
                    mask_feats = self.mask_branch[i](mask_feats)
                    times['mask_branch'].append(time.time() - start)
                else:
                    start = time.time()
                    mask_feats = self.mask_branch[i](x)
                    times['mask_branch'].append(time.time() - start)

                # Timing inst_feats_concat
                if i != 0:
                    start = time.time()
                    inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
                    inst_feats = torch.cat([x, inst_feats], dim=1)
                    times['inst_feats_concat'].append(time.time() - start)

                    start = time.time()
                    coord_features = self.compute_coordinates(inst_feats)
                    times['coord'].append(time.time() - start)

                    start = time.time()
                    inst_feats = torch.cat([coord_features, inst_feats], dim=1)
                    inst_feats = self.instance_branch[i](inst_feats)
                    times['inst_branch'].append(time.time() - start)
                else:
                    start = time.time()
                    coord_features = self.compute_coordinates(x)
                    times['coord'].append(time.time() - start)

                    start = time.time()
                    inst_feats = torch.cat([coord_features, x], dim=1)
                    inst_feats = self.instance_branch[i](inst_feats)
                    times['inst_branch'].append(time.time() - start)
            else:
                if i == self.n_levels - 1:
                    start = time.time()
                    mask_feats = self.mask_branch[0](x)
                    times['mask_branch'].append(time.time() - start)

                    start = time.time()
                    coord_features = self.compute_coordinates(x)
                    times['coord'].append(time.time() - start)

                    start = time.time()
                    inst_feats = torch.cat([coord_features, x], dim=1)
                    inst_feats = self.instance_branch[0](inst_feats)
                    times['inst_branch'].append(time.time() - start)
        
        # Timing instance head
        start = time.time()
        results = self.instance_head(inst_feats)
        
        mask_feats = self.projection(mask_feats)
        times['instance_head'].append(time.time() - start)

        # Print the mean times for each part
        for key in times:
            if times[key]:  # Check if the list is not empty
                print(f"Average {key.replace('_', ' ').capitalize()} Time: {sum(times[key]):.6f} seconds")
            else:
                print(f"No data collected for {key.replace('_', ' ').capitalize()}")


        return results, mask_feats