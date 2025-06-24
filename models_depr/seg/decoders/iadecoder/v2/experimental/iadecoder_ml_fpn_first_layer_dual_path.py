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

from models.seg.decoders.base import BaseDecoder
from configs.structure import Decoder
from utils.registry import DECODERS, HEADS
from omegaconf import OmegaConf

from models.seg.nn.blocks import MLP, FFNLayer
from models.seg.nn.blocks import CrossAttentionLayer, SelfAttentionLayer, PositionEmbeddingSine
from models.seg.heads.common import _make_stack_3x3_convs


@DECODERS.register(name='iadecoder_ml_fpn/experimental/first_layer_dual_path')
class IADecoder(BaseDecoder):
    def __init__(self, 
                 cfg: Decoder, 
                 embed_dims: list = [], 
                 n_levels: int = 4
                 ):
        super(BaseDecoder, self).__init__()

        self.n_levels = n_levels - 1
        num_convs = cfg.mask_branch.num_convs
        mask_dim = cfg.mask_branch.dim

        hidden_dim = cfg.hidden_dim
        num_classes = cfg.num_classes
        num_queries = cfg.num_queries
        dim_feedforward = cfg.dim_feedforward
        nheads = cfg.nheads
        dropout = cfg.dropout
        pre_norm = cfg.pre_norm
        self.num_layers = cfg.dec_layers + self.n_levels - 1
        self.dec_layers = cfg.dec_layers
        self.embed_dims = embed_dims
        self.semantic_ce_loss = True

        embed_dims = self.embed_dims[::-1]

        self.skip_conv_layers = nn.ModuleList([])
        for i in range(self.n_levels):
            in_channels = embed_dims[i]
            skip_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
            self.skip_conv_layers.append(skip_conv)

        # upconv layers.
        self.up_conv_layers = nn.ModuleList([])
        for i in range(self.n_levels):
            in_channels = hidden_dim + 2 if i == 0 else hidden_dim * 2 + 2
            upconv = nn.Sequential( 
                DoubleConv_v2(in_channels, hidden_dim), 
                SE_block(num_features=hidden_dim) 
            )
            self.up_conv_layers.append(upconv)


        embed_dims = embed_dims[1:] + [embed_dims[-1]]

        # mask branch.
        self.mask_branch = nn.ModuleList([])
        mask_branch_layer = HEADS.get(cfg.mask_branch.type)
        for i in range(self.n_levels):
            mask_branch = mask_branch_layer(
                    in_channels=hidden_dim, 
                    out_channels=mask_dim, 
                    num_convs=num_convs
                )
            self.mask_branch.append(mask_branch)

        # mask head.
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


        # transformer layers for updating pixel features.
        # ca.
        self.transformer_pixel_cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dropout=dropout,
                normalize_before=pre_norm,
            )
            for _ in range(self.n_levels)
        ])
    
        # ffn.
        self.transformer_pixel_ffn_layers = nn.ModuleList([
            FFNLayer(
                d_model=hidden_dim,
                dim_feedforward=dim_feedforward, 
                dropout=dropout, 
                normalize_before=pre_norm
            )
            for _ in range(self.n_levels)
        ])


        # learnable query features.
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        if self.semantic_ce_loss:
            self.class_embed = nn.Linear(hidden_dim, num_classes+1)
        else:
            self.class_embed = nn.Linear(hidden_dim, num_classes)
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

        # c2_xavier_fill(self.lateral_conv[0])
        # c2_xavier_fill(self.output_conv[0])

        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        init.normal_(self.class_embed.weight, std=0.01)
        init.constant_(self.class_embed.bias, bias_value)
        init.normal_(self.mask_embed.weight, std=0.01)
        init.constant_(self.mask_embed.bias, 0.0)
        

    def forward_dual_path_layer(self, pixel_feats, query_feat, query_embed, pos, layer_idx):      
        # pixel ca.
        pixel_feats, pixel_ca_attn = self.transformer_pixel_cross_attention_layers[layer_idx - self.dec_layers + 1](
            pixel_feats, query_feat, 
            memory_mask=None, 
            memory_key_padding_mask=None, 
            pos=query_embed, query_pos=pos
            )
        
        # pixel ffn.
        pixel_feats = self.transformer_pixel_ffn_layers[layer_idx - self.n_levels + 1](pixel_feats)


        # query ca.
        query_feat, query_ca_attn = self.transformer_instance_cross_attention_layers[layer_idx](
            query_feat, pixel_feats, 
            memory_mask=None, 
            memory_key_padding_mask=None, 
            pos=pos, query_pos=query_embed
            )
        
        # query sa.
        query_feat, query_sa_attn = self.transformer_instance_self_attention_layers[layer_idx](
            query_feat, tgt_mask=None, 
            tgt_key_padding_mask=None, 
            query_pos=query_embed
            )
        
        # query ffn.
        query_feat = self.transformer_instance_ffn_layers[layer_idx](query_feat)

        # attn.
        HW, B, _ = pixel_feats.shape
        H, W = int(HW ** 0.5), int(HW ** 0.5)
        pixel_ca_attn = pixel_ca_attn.permute(0, 2, 1).view(B, -1, H, W)
        query_ca_attn = query_ca_attn.view(B, -1, H, W)
        query_sa_attn = query_sa_attn.permute(0, 2, 1)

        attn = {
            f'pixel_ca_attn.{layer_idx}': pixel_ca_attn,
            f'query_ca_attn.{layer_idx}': query_ca_attn,
            f'query_sa_attn.{layer_idx}': query_sa_attn
        }
        
        return pixel_feats, query_feat, attn


    def forward_one_layer(self, pixel_feats, query_feat, query_embed, pos, layer_idx):      
        # query ca.
        query_feat, query_ca_attn = self.transformer_instance_cross_attention_layers[layer_idx](
            query_feat, pixel_feats, 
            memory_mask=None, 
            memory_key_padding_mask=None, 
            pos=pos, query_pos=query_embed
            )
        
        # query sa.
        query_feat, query_sa_attn = self.transformer_instance_self_attention_layers[layer_idx](
            query_feat, tgt_mask=None, 
            tgt_key_padding_mask=None, 
            query_pos=query_embed
            )
        
        # query ffn.
        query_feat = self.transformer_instance_ffn_layers[layer_idx](query_feat)

    
        # attn.
        HW, B, _ = pixel_feats.shape
        H, W = int(HW ** 0.5), int(HW ** 0.5)
        query_ca_attn = query_ca_attn.view(B, -1, H, W)
        query_sa_attn = query_sa_attn.permute(0, 2, 1)

        attn = {
            f'query_ca_attn.{layer_idx}': query_ca_attn,
            f'query_sa_attn.{layer_idx}': query_sa_attn
        }
        
        return pixel_feats, query_feat, attn
        

    def _forward(self, features):
        # features - multi-scale encoder features (1/4, 1/8, 1/16, 1/32)
        B, C, H, W = features[0].shape

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat(1, B, 1)
        
        # pixel decoder.
        predictions_mask = []
        predictions_class = []
        predictions_boxes = []
        src = []
        predictions_attn = []
        interm_queries = []
        for i in range(self.n_levels):
            # upconvs.
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


            # decoupling.
            if i != 0:
                mask_feats = nn.UpsamplingBilinear2d(scale_factor=2)(mask_feats)
                mask_feats = mask_feats + x
                mask_feats = self.mask_branch[i](mask_feats)   
            else:
                mask_feats = self.mask_branch[i](x)


            # transformer decoder.
            if i == self.n_levels - 1:
                for _, j in enumerate(range(self.dec_layers)):
                    layer_idx = self.n_levels - 1 + j

                    B, C, H, W = mask_feats.shape
                    pos = self.pe_layer(mask_feats, None)
                    pos = pos.flatten(2).permute(2, 0, 1)
                    mask_feats = mask_feats.flatten(2).permute(2, 0, 1)
                    
                    mask_feats, query_feat, attn = self.forward_last_layer(mask_feats, query_feat, query_embed, pos, layer_idx)
                    
                    mask_feats = mask_feats.permute(1, 2, 0).view(B, C, H, W)

                    interm_queries.append(query_feat)
                    predictions_attn.append(attn)
            else:
                layer_idx = i

                B, C, H, W = mask_feats.shape
                pos = self.pe_layer(mask_feats, None)
                pos = pos.flatten(2).permute(2, 0, 1)
                mask_feats = mask_feats.flatten(2).permute(2, 0, 1)
                
                mask_feats, query_feat, attn = self.forward_one_layer(mask_feats, query_feat, query_embed, pos, layer_idx)
                
                mask_feats = mask_feats.permute(1, 2, 0).view(B, C, H, W)

                interm_queries.append(query_feat)
                predictions_attn.append(attn)


        # getting the first backbone feature map (1/4 - 'res2')
        # output_conv( lateral_conv(features[0]) + (x2)mask_feats )
        # src[-1] - 1/8
        out = features[0] # 1/4
        y = self.lateral_conv(out) + F.interpolate(mask_feats, size=out.shape[-2:], 
                                                   mode="bilinear", align_corners=False)
        mask_features = self.output_conv(y)


        for i, _query_feat in enumerate(interm_queries):
            # adding aux outputs.
            outputs_class, outputs_mask, outputs_boxes = \
                self.forward_prediction_heads(_query_feat, mask_features)
            
            predictions_mask.append(outputs_mask)
            predictions_class.append(outputs_class)
            predictions_boxes.append(outputs_boxes)
                

        results = {
            'logits': predictions_class[-1],
            'bboxes': predictions_boxes[-1],
            'masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class[:-1], predictions_mask[:-1], predictions_boxes[:-1]
            )
        }

        results["mask_feats"] = mask_features
        results["attn"] = predictions_attn
    
        return results
    

    def forward_prediction_heads(self, output, mask_features):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        # predictions.
        pred_logits = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        pred_bboxes = self.bbox_pred(decoder_output)
    
        # instance masks.
        N = mask_embed.shape[1]
        B, C, H, W = mask_features.shape
        outputs_mask = torch.bmm(mask_embed, mask_features.view(B, C, H * W))
        outputs_mask = outputs_mask.view(B, N, H, W)
        # outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        outputs_boxes = pred_bboxes.sigmoid()

        return pred_logits, outputs_mask, outputs_boxes


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, out_boxes=None):
        return [
            {"pred_logits": a, "pred_instance_masks": b, "pred_bboxes": c}
            for a, b, c in zip(outputs_class, outputs_seg_masks, out_boxes)
        ]
    

    def process_outputs(self, results, ori_shape):
        logits = results.get('logits')
        scores = results.get('objectness_scores')
        inst_masks = results.get('masks')
        bboxes = results.get('bboxes')
        mask_feats = results.get('mask_feats')
        iams = results.get('iams')
        attn = results.get('attn')
        aux_outputs = results.get('aux_outputs')
        
        # instance masks.
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
            }, 
            'attn': attn,
            'aux_outputs': aux_outputs
        }
    
        return output

