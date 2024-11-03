import torch
from torch import nn

import sys
sys.path.append("./")

from models.seg.decoders.iadecoder.iadecoder import IADecoder
from configs.structure import Decoder
from utils.registry import HEADS, DECODERS

from models.seg.nn.blocks import (DoubleConv, DoubleConv_v1, DoubleConv_v2, 
                                   SE_block)
from omegaconf import OmegaConf


@DECODERS.register(name='iadecoder_ml_light')
class IADecoder(IADecoder):
    def __init__(self, cfg: Decoder, embed_dims: list = [], n_levels: int = 4):
        super().__init__(cfg, embed_dims, n_levels)

        self.n_levels = n_levels - 1

        self.bridge = nn.Sequential(
            DoubleConv_v2(embed_dims[-1], embed_dims[-2]),
        )

        embed_dims = embed_dims[:-1][::-1]
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
        mask_dim = cfg.mask_branch.dim
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

        self.projection = nn.Conv2d(mask_dim, self.kernel_dim, kernel_size=1)
        
        # instance branch.
        self.instance_branch = nn.ModuleList([])
        instance_branch_layer = HEADS.get(cfg.instance_branch.type)
        for i in range(self.n_levels):
            if i == 0:
                instance_branch = instance_branch_layer(
                    in_channels=embed_dims[i] + 2, 
                    out_channels=self.inst_dim, 
                    num_convs=self.num_convs
                )
            else:
                instance_branch = instance_branch_layer(
                    in_channels=embed_dims[i] + 2, 
                    out_channels=self.inst_dim, 
                    num_convs=self.num_convs
                )
            self.instance_branch.append(instance_branch)

        # instance head.
        self.instance_head = nn.ModuleList([])
        for i in range(self.n_levels):
            instance_head = OmegaConf.to_container(cfg.instance_head, resolve=True)
            instance_head['in_res'] = (16 * (2 ** (i+1)), 16 * (2 ** (i+1)))

            instance_head = HEADS.build(instance_head)
            self.instance_head.append(instance_head)

        self._init_weights()
    

    def _forward(self, skips):
        x = self.bridge(skips[-1])
        skips = skips[:-1]

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


            # if i != 0:
            #     inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
            #     inst_feats = torch.cat([x, inst_feats], dim=1)

            #     coord_features = self.compute_coordinates(inst_feats)
            #     inst_feats = torch.cat([coord_features, inst_feats], dim=1)
            #     inst_feats = self.instance_branch[i](inst_feats)
            # else:
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
            # attn_mask = results['attn_mask']

    
        mask_feats = self.projection(mask_feats)
        results["mask_feats"] = mask_feats
        results["inst_feats"] = inst_feats
        # results["attn_mask"] = attn_mask
    
        return results
    



# import torch
# from torch import nn

# import sys
# sys.path.append("./")

# from models.seg.decoders.iadecoder.iadecoder import IADecoder
# from configs.structure import Decoder
# from utils.registry import HEADS, DECODERS
# from omegaconf import OmegaConf


# @DECODERS.register(name='iadecoder_ml_light')
# class IADecoder(IADecoder):
#     def __init__(self, cfg: Decoder, embed_dims: list = [], n_levels: int = 4):
#         super().__init__(cfg, embed_dims, n_levels)

#         embed_dims = self.embed_dims[::-1]
#         embed_dims = embed_dims[1:] + [embed_dims[-1]]

#         # mask branch.
#         mask_branch_layer = HEADS.get(cfg.mask_branch.type)
#         self.mask_branch = nn.ModuleList([])
#         for i in range(self.n_levels):
#             if i == 0:
#                 mask_branch = mask_branch_layer(
#                     in_channels=embed_dims[i], 
#                     out_channels=self.mask_dim, 
#                     num_convs=self.num_convs
#                 )
#             else:
#                 mask_branch = mask_branch_layer(
#                     in_channels=embed_dims[i] + self.mask_dim, 
#                     out_channels=self.mask_dim, 
#                     num_convs=self.num_convs
#                 )
#             self.mask_branch.append(mask_branch)
        
#         # instance branch.
#         self.instance_branch = nn.ModuleList([])
#         instance_branch_layer = HEADS.get(cfg.instance_branch.type)
#         for i in range(self.n_levels):
#             if i == 0:
#                 instance_branch = instance_branch_layer(
#                     in_channels=embed_dims[i] + 2, 
#                     out_channels=self.inst_dim, 
#                     num_convs=self.num_convs
#                 )
#             else:
#                 instance_branch = instance_branch_layer(
#                     in_channels=embed_dims[i] + 2, 
#                     out_channels=self.inst_dim, 
#                     num_convs=self.num_convs
#                 )
#             self.instance_branch.append(instance_branch)

#         # instance head.
#         self.instance_head = nn.ModuleList([])
#         for i in range(self.n_levels):
            
#             instance_head = OmegaConf.to_container(cfg.instance_head, resolve=True)
#             instance_head['in_res'] = (16 * (2 ** i), 16 * (2 ** i))

#             instance_head = HEADS.build(instance_head)
#             self.instance_head.append(instance_head)

#         self._init_weights()
    

#     def _forward(self, skips):
#         for i in range(self.n_levels):
#             if i != 0:
#                 coord_features = self.compute_coordinates(x)
#                 x = torch.cat([coord_features, x], dim=1)
                
#                 x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
#                 x = torch.cat([x, skips[-(i + 1)]], dim=1)
#                 x = self.up_conv_layers[i](x)
#             else:
#                 coord_features = self.compute_coordinates(skips[-1])
#                 x = torch.cat([coord_features, skips[-1]], dim=1)
#                 x = self.up_conv_layers[i](x)

            
#             if i != 0:
#                 mask_feats = nn.UpsamplingBilinear2d(scale_factor=2)(mask_feats)
#                 mask_feats = torch.cat([x, mask_feats], dim=1)
#                 mask_feats = self.mask_branch[i](mask_feats)   
#             else:
#                 mask_feats = self.mask_branch[i](x)


#             coord_features = self.compute_coordinates(x)
#             inst_feats = torch.cat([coord_features, x], dim=1)
#             inst_feats = self.instance_branch[i](inst_feats)


#             if i != 0:
#                 results = self.instance_head[i](inst_feats, mask_feats, inst_embed)
#                 inst_embed = results["inst_feats"]['instance_feats']
#             else:
#                 results = self.instance_head[i](inst_feats, mask_feats)
#                 inst_embed = results["inst_feats"]['instance_feats']

#             mask_feats = results['mask_pixel_feats']
#             inst_feats = results['inst_pixel_feats']
#             # attn_mask = results['attn_mask']

    
#         mask_feats = self.projection(mask_feats)
#         results["mask_feats"] = mask_feats
#         results["inst_feats"] = inst_feats
#         # results["attn_mask"] = attn_mask
    
#         return results
    


        