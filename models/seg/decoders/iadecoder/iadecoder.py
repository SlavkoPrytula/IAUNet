import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn.weight_init import c2_msra_fill
from torch.nn import init

import sys
sys.path.append("./")

from models.seg.nn.blocks import (DoubleConv, DoubleConv_v1, DoubleConv_v2, 
                                   SE_block)

from models.seg.decoders.base import BaseDecoder
from configs.structure import Decoder
from utils.registry import DECODERS, HEADS
from omegaconf import OmegaConf


@DECODERS.register(name='iadecoder')
class IADecoder(BaseDecoder):
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
        embed_dims = embed_dims + [embed_dims[-1]]
        
        self.up_conv_layers = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                in_channels = embed_dims[i] + 2
            else:
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
                    in_channels=embed_dims[i] + self.inst_dim + 2, 
                    out_channels=self.inst_dim, 
                    num_convs=self.num_convs
                )
            self.instance_branch.append(instance_branch)

        instance_head = OmegaConf.to_container(cfg.instance_head, resolve=True)
        instance_head['in_res'] = (128, 128)

        # instance head.
        self.instance_head = HEADS.build(instance_head)

        self._init_weights()


    def _init_weights(self):
        for modules in [self.up_conv_layers]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

        c2_msra_fill(self.projection)


    def forward(self, skips, ori_shape):
        """
        Default decoder forward function:
        - _forward() -> dict()
        - process_outputs() -> dict()

        ori_shape - max_shape of an image to be returned
        """
        results = self._forward(skips)
        results = self.process_outputs(results, ori_shape)
        return results
    

    def _forward(self, skips):
        for i in range(self.n_levels):
            if i != 0:
                coord_features = self.compute_coordinates(x)
                x = torch.cat([coord_features, x], dim=1)
                
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
                x = torch.cat([x, skips[-(i + 1)]], dim=1)
                x = self.up_conv_layers[i](x)
            else:
                coord_features = self.compute_coordinates(skips[-1])
                x = torch.cat([coord_features, skips[-1]], dim=1)
                x = self.up_conv_layers[i](x)


            if not self.last_layer_only:
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
            else:
                if i == self.n_levels - 1:
                    mask_feats = self.mask_branch[0](x)
                    coord_features = self.compute_coordinates(x)
                    inst_feats = torch.cat([coord_features, x], dim=1)
                    inst_feats = self.instance_branch[0](inst_feats)
        


        results = self.instance_head(inst_feats)
        mask_feats = self.projection(mask_feats)
        results["mask_feats"] = mask_feats
        results["inst_feats"] = inst_feats
    
        return results
    

    def process_outputs(self, results, ori_shape):
        logits = results["logits"]
        scores = results["objectness_scores"]
        inst_kernel = results["kernels"]["instance_kernel"]
        bboxes = results["bboxes"]['instance_bboxes']
        mask_feats = results["mask_feats"]
        inst_feats = results["inst_feats"]
        # attn_mask = results.get('attn_mask')
        
        # instance masks.
        N = inst_kernel.shape[1]
        B, C, H, W = mask_feats.shape

        inst_masks = torch.bmm(inst_kernel, mask_feats.view(B, C, H * W))
        inst_masks = inst_masks.view(B, N, H, W)
        bboxes = bboxes.sigmoid()

        # nhead = attn_mask.shape[0] // B
        # h = w = int(attn_mask.shape[2] ** 0.5)
        # attn_mask = attn_mask.view(B, nhead, N, h, w)

        inst_masks = F.interpolate(inst_masks, size=ori_shape, 
                                   mode="bilinear", align_corners=False)

        output = {
            'pred_logits': logits,
            'pred_scores': scores,
            'pred_iams': results['iams'],
            'pred_instance_masks': inst_masks,
            'pred_bboxes': bboxes,
            'pred_instance_feats': {
                "mask_feats": mask_feats,
                "inst_feats": inst_feats,
                # "attn_mask": attn_mask
            }
        }
    
        return output