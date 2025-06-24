import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill
from torch.nn import init
import numpy as np 

import sys
sys.path.append("./")

from models.seg.nn.blocks import (DoubleConv, DoubleConv_v1, DoubleConv_v2, 
                                  SE_block)

from models.seg.decoders.iadecoder.iadecoder import IADecoder
from models.seg.decoders.base import BaseDecoder
from configs.structure import Decoder
from utils.registry import DECODERS, HEADS
from omegaconf import OmegaConf

from models.seg.nn.blocks import MLP, FFNLayer
from models.seg.nn.blocks import CrossAttentionLayer, SelfAttentionLayer, PositionEmbeddingSine



@DECODERS.register(name='iadecoder_ml_fpn_aux')
class IADecoder(BaseDecoder):
    def __init__(self, 
                 cfg: Decoder, 
                 embed_dims: list = [], 
                 n_levels: int = 4
                 ):
        super(BaseDecoder, self).__init__()

        self.n_levels = n_levels
        num_convs = cfg.num_convs
        mask_dim = cfg.mask_branch.dim
        inst_dim = cfg.instance_branch.dim

        hidden_dim = cfg.hidden_dim
        num_classes = cfg.num_classes + 1
        num_queries = cfg.num_queries
        dim_feedforward = cfg.dim_feedforward
        nheads = cfg.nheads
        dropout = cfg.dropout
        pre_norm = cfg.pre_norm
        self.num_layers = cfg.dec_layers * n_levels

        self.embed_dims = embed_dims

        embed_dims = self.embed_dims[::-1]
        fpn_dim = 256

        self.skip_conv_layers = nn.ModuleList([])
        for i in range(n_levels - 1):
            skip_in_channels = embed_dims[i]
            skip_out_channels = fpn_dim

            skip_conv = nn.Conv2d(skip_in_channels, skip_out_channels, kernel_size=1)
            self.skip_conv_layers.append(skip_conv)

        
        self.up_conv_layers = nn.ModuleList([])
        for i in range(n_levels - 1):
            if i == 0:
                in_channels = fpn_dim + 2
            else:
                in_channels = fpn_dim * 2 + 2
            out_channels = fpn_dim

            upconv = nn.Sequential( 
                DoubleConv_v2(in_channels, out_channels), 
                SE_block(num_features=out_channels) 
            )
            self.up_conv_layers.append(upconv)


        embed_dims = embed_dims[1:] + [embed_dims[-1]]

        # mask branch.
        mask_branch_layer = HEADS.get(cfg.mask_branch.type)
        self.mask_branch = nn.ModuleList([])
        for i in range(n_levels - 1):
            if i == 0:
                mask_branch = mask_branch_layer(
                    in_channels=fpn_dim, 
                    out_channels=mask_dim, 
                    num_convs=num_convs
                )
            else:
                mask_branch = mask_branch_layer(
                    in_channels=fpn_dim, 
                    out_channels=mask_dim, 
                    num_convs=num_convs
                )
            self.mask_branch.append(mask_branch)
        
        # instance features.
        self.instance_branch = nn.ModuleList([])
        instance_branch_layer = HEADS.get(cfg.instance_branch.type)
        for i in range(n_levels - 1):
            if i == 0:
                instance_branch = instance_branch_layer(
                    in_channels=fpn_dim + 2, 
                    out_channels=inst_dim, 
                    num_convs=num_convs
                )
            else:
                instance_branch = instance_branch_layer(
                    in_channels=fpn_dim + 2, 
                    out_channels=inst_dim, 
                    num_convs=num_convs
                )
            self.instance_branch.append(instance_branch)


        self.lateral_conv = nn.Sequential(
            nn.Conv2d(mask_dim, mask_dim, kernel_size=1),
            nn.BatchNorm2d(mask_dim),
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(mask_dim, mask_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mask_dim),
            nn.ReLU(inplace=True),
        )


        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # ca.
        self.transformer_instance_cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dropout=dropout,
                normalize_before=pre_norm,
            )
            for _ in range(self.num_layers)
        ])
        
        # sa.
        self.transformer_instance_self_attention_layers = nn.ModuleList([
            SelfAttentionLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dropout=dropout,
                normalize_before=pre_norm,
            )
            for _ in range(self.num_layers)
        ])

        # ffn.
        self.transformer_instance_ffn_layers = nn.ModuleList([
            FFNLayer(
                d_model=hidden_dim,
                dim_feedforward=dim_feedforward, 
                dropout=dropout, 
                normalize_before=pre_norm
            )
            for _ in range(self.num_layers)
        ])

        # learnable query features.
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.cls_score = nn.Linear(hidden_dim, num_classes)
        self.mask_embed =  nn.Linear(hidden_dim, mask_dim) #MLP(hidden_dim, hidden_dim, mask_dim, 3) 
        self.objectness = nn.Linear(hidden_dim, 1)
        self.bbox_pred = nn.Linear(hidden_dim, 4)
        
        self.prior_prob = 0.01
    
        self._init_weights()


    def _init_weights(self):
        for modules in [self.up_conv_layers, self.skip_conv_layers]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, bias_value)
        init.normal_(self.mask_embed.weight, std=0.01)
        init.constant_(self.mask_embed.bias, 0.0)
        

    def _forward(self, features):
        # features - multi-scale encoder features (1/4, 1/8, 1/16, 1/32)
        
        B, C, H, W = features[0].shape

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat(1, B, 1)
        
        aux_outputs = []
        for i in range(self.n_levels - 1):
            if i != 0:
                coord_features = self.compute_coordinates(x)
                x = torch.cat([coord_features, x], dim=1)
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)

                skip = features[-(i + 1)]
                skip = self.skip_conv_layers[i](skip)

                x = torch.cat([x, skip], dim=1)
                x = self.up_conv_layers[i](x)
            else:
                skip = features[-1]
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

            # transformer decoder.
            B, C, H, W = mask_feats.shape

            pos = self.pe_layer(mask_feats, None)
            pos = pos.flatten(2).permute(2, 0, 1)
            mask_feats = mask_feats.flatten(2).permute(2, 0, 1)

            # query ca.
            query_feat, _ = self.transformer_instance_cross_attention_layers[i](
                query_feat, mask_feats, 
                memory_mask=None, 
                memory_key_padding_mask=None, 
                pos=pos, query_pos=query_embed
                )
            
            # query sa.
            query_feat, _ = self.transformer_instance_self_attention_layers[i](
                query_feat, tgt_mask=None, 
                tgt_key_padding_mask=None, 
                query_pos=query_embed
                )
            
            # query ffn.
            query_feat = self.transformer_instance_ffn_layers[i](query_feat)
            
            mask_feats = mask_feats.permute(1, 2, 0).view(B, C, H, W)


            # aux_output = self._set_aux_loss(results)
            # aux_outputs.append(aux_output)

        # getting the first backbone feature map (1/4 - 'res2')
        # output_conv( lateral_conv(features[0]) + (x2)mask_feats ) 
        out = features[0]
        y = self.lateral_conv(out) + F.interpolate(mask_feats, size=out.shape[-2:], 
                                                   mode="bilinear", align_corners=False)
        mask_feats = self.output_conv(y)




        query_feat = self.decoder_norm(query_feat)
        query_feat = query_feat.transpose(0, 1)

        # predictions.
        pred_logits = self.cls_score(query_feat)
        mask_embed = self.mask_embed(query_feat)
        pred_scores = self.objectness(query_feat)
        pred_bboxes = self.bbox_pred(query_feat)

        results = {
            'logits': pred_logits,
            'objectness_scores': pred_scores,
            'bboxes': {
                'instance_bboxes': pred_bboxes
            },
            'mask_embed': mask_embed,
        }

        results["aux_outputs"] = aux_outputs

        results["mask_feats"] = mask_feats
        results["inst_feats"] = inst_feats
    
        return results
    

    def forward_prediction_heads(self, query_feat, mask_feats):
        query_feat = self.decoder_norm(query_feat)
        query_feat = query_feat.transpose(0, 1)

        # predictions.
        pred_logits = self.cls_score(query_feat)
        mask_embed = self.mask_embed(query_feat)
        pred_scores = self.objectness(query_feat)
        pred_bboxes = self.bbox_pred(query_feat)

        results = {
            'logits': pred_logits,
            'objectness_scores': pred_scores,
            'bboxes': {
                'instance_bboxes': pred_bboxes
            },
            'mask_embed': mask_embed,
        }

        return results
    

    def process_outputs(self, results, ori_shape):
        logits = results["logits"]
        scores = results["objectness_scores"]
        mask_embed = results["mask_embed"]
        bboxes = results["bboxes"]['instance_bboxes']
        mask_feats = results["mask_feats"]
        inst_feats = results["inst_feats"]
        iams = results.get('iams')
        inst_pixel_attn = results.get('inst_pixel_attn')
        mask_pixel_attn = results.get('mask_pixel_attn')
        query_sa_attn = results.get('query_sa_attn')
        
        # instance masks.
        N = mask_embed.shape[1]
        B, C, H, W = mask_feats.shape

        # inst_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feats)

        inst_masks = torch.bmm(mask_embed, mask_feats.view(B, C, H * W))
        inst_masks = inst_masks.view(B, N, H, W)
        bboxes = bboxes.sigmoid()

        inst_masks = F.interpolate(inst_masks, size=ori_shape, 
                                   mode="bilinear", align_corners=False)

        output = {
            'pred_logits': logits,
            'pred_scores': scores,
            'pred_iams': iams,
            'pred_instance_masks': inst_masks,
            'pred_bboxes': bboxes,
            'pred_instance_feats': {
                "mask_feats": mask_feats,
                "inst_feats": inst_feats,
            }, 
            'attn': {
                "inst_pixel_attn": inst_pixel_attn,
                "mask_pixel_attn": mask_pixel_attn, 
                "query_sa_attn": query_sa_attn
            }
        }
    
        return output


    def _set_aux_loss(self, results):
        logits = results["logits"]
        scores = results["objectness_scores"]
        bboxes = results["bboxes"]['instance_bboxes'].sigmoid()
        inst_masks = results["masks"]["instance_masks"]

        output = {
            'pred_logits': logits,
            'pred_scores': scores,
            'pred_iams': results['iams'],
            'pred_instance_masks': inst_masks,
            'pred_bboxes': bboxes,
        }
    
        return output
    
    


# for j in range(self.num_layers // self.n_levels):
#     # get idx of dec. layer
#     layer_idx = i * self.num_layers // self.n_levels + j

#     # query ca.
#     query_feat, _ = self.transformer_instance_cross_attention_layers[layer_idx](
#         query_feat, inst_feats, 
#         memory_mask=None, 
#         memory_key_padding_mask=None, 
#         pos=pos, query_pos=query_embed
#         )
    
#     # query sa.
#     query_feat, _ = self.transformer_instance_self_attention_layers[layer_idx](
#         query_feat, tgt_mask=None, 
#         tgt_key_padding_mask=None, 
#         query_pos=query_embed
#         )
    
#     # query ffn.
#     query_feat = self.transformer_instance_ffn_layers[layer_idx](query_feat)