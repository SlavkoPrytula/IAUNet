import torch
from torch import le, nn
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



@DECODERS.register(name='iadecoder_ml_fpn_dual_path_staged')
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
        num_classes = cfg.num_classes + 1
        num_queries = cfg.num_queries
        dim_feedforward = cfg.dim_feedforward
        nheads = cfg.nheads
        dropout = cfg.dropout
        pre_norm = cfg.pre_norm

        self.num_layers = cfg.dec_layers * self.n_levels
        self.dec_layers = cfg.dec_layers
        self.embed_dims = embed_dims

        embed_dims = self.embed_dims[::-1]

        self.skip_conv_layers = nn.ModuleList([])
        for i in range(self.n_levels):
            in_channels = embed_dims[i]
            skip_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
            self.skip_conv_layers.append(skip_conv)

        
        self.up_conv_layers = nn.ModuleList([])
        for i in range(self.n_levels):
            in_channels = hidden_dim + 2 if i == 0 else hidden_dim * 2 + 2
            upconv = nn.Sequential( 
                DoubleConv_v2(in_channels, hidden_dim), 
                SE_block(num_features=hidden_dim) 
            )
            self.up_conv_layers.append(upconv)


        embed_dims = embed_dims[1:] + [embed_dims[-1]]


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
        

        # transformer decoder.
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # transformer layers for updating queries.
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
        # sa.
        self.transformer_pixel_self_attention_layers = nn.ModuleList([
            SelfAttentionLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dropout=dropout,
                normalize_before=pre_norm,
            )
            for _ in range(self.num_layers)
        ])

        # ca.
        self.transformer_pixel_cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dropout=dropout,
                normalize_before=pre_norm,
            )
            for _ in range(self.num_layers)
        ])

        # ffn.
        self.transformer_pixel_ffn_layers = nn.ModuleList([
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
        self.mask_embed =  nn.Linear(hidden_dim, mask_dim) # MLP(hidden_dim, hidden_dim, mask_dim, 3) 
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


    def forward_one_layer(self, pixel_feats, query_feat, query_embed, pos, layer_idx):        
        # query sa.
        query_feat, _ = self.transformer_instance_self_attention_layers[layer_idx](
            query_feat, tgt_mask=None, 
            tgt_key_padding_mask=None, 
            query_pos=query_embed
            )
        
        # query ca.
        query_feat, _ = self.transformer_instance_cross_attention_layers[layer_idx](
            query_feat, pixel_feats, 
            memory_mask=None, 
            memory_key_padding_mask=None, 
            pos=pos, query_pos=query_embed
            )
        
        # query ffn.
        query_feat = self.transformer_instance_ffn_layers[layer_idx](query_feat)

        # pixel ca.
        pixel_feats, _ = self.transformer_pixel_cross_attention_layers[layer_idx](
            pixel_feats, query_feat, 
            memory_mask=None, 
            memory_key_padding_mask=None, 
            pos=query_embed, query_pos=pos
            )
        
        # pixel sa.
        # pixel_feats, _ = self.transformer_pixel_self_attention_layers[layer_idx](
        #     pixel_feats, tgt_mask=None, 
        #     tgt_key_padding_mask=None, 
        #     query_pos=pos
        #     )
        
        # pixel ffn.
        pixel_feats = self.transformer_pixel_ffn_layers[layer_idx](pixel_feats)

        return pixel_feats, query_feat
        

    def _forward(self, features):
        # features - multi-scale encoder features (1/4, 1/8, 1/16, 1/32)
        B, C, H, W = features[0].shape

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat(1, B, 1)
        
        src = []
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

            src.append(x)
    
        # here we want to take the intermediate feature maps from the pixel decoder and refine them 
        # using the transformer decoder.
        # for every multi-scale feature map we fill process it with a single transformer layer with the queries
        # each tranformer layer updates the query and pixel features.
        # we will store the refined pixel features in a list.
        # after the loop we will use the refined pixel features and loop again for num_layers 
        # (e.g. num_layers=3 means we do 3 update loops over pixel features).

        for _, j in enumerate(range(self.dec_layers)):
            for i, x in enumerate(src):
                layer_idx = j * self.n_levels + i

                # transformer decoder.
                B, C, H, W = x.shape

                pos = self.pe_layer(x, None)
                pos = pos.flatten(2).permute(2, 0, 1)
                x = x.flatten(2).permute(2, 0, 1)
                
                x, query_feat = self.forward_one_layer(x, query_feat, query_embed, pos, layer_idx)
                
                x = x.permute(1, 2, 0).view(B, C, H, W)
                src[i] = x

        # we can get the mask_features here and the use duap-path to update them and queries.
        # <>

        # predictions_mask = []
        # predictions_class = []
        # predictions_boxes = []
        # for i, x in enumerate(src):
        #     # adding aux outputs.
        #     outputs_class, outputs_mask, outputs_boxes = \
        #         self.forward_prediction_heads(query_feat, x)
            
        #     predictions_mask.append(outputs_mask)
        #     predictions_class.append(outputs_class)
        #     predictions_boxes.append(outputs_boxes)


        # getting the first backbone feature map (1/4 - 'res2')
        # output_conv( lateral_conv(features[0]) + (x2)mask_feats )
        # src[-1] - 1/8
        out = features[0] # 1/4
        y = self.lateral_conv(out) + F.interpolate(src[-1], size=out.shape[-2:], 
                                                   mode="bilinear", align_corners=False)
        mask_features = self.output_conv(y)
        

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
            # 'aux_outputs': self._set_aux_loss(
            #     predictions_class, predictions_mask, predictions_boxes
            # )
        }

        results["mask_feats"] = mask_features
    
        return results
    

    def forward_prediction_heads(self, output, mask_features):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        # predictions.
        pred_logits = self.cls_score(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        pred_bboxes = self.bbox_pred(decoder_output)

        # outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        N = mask_embed.shape[1]
        B, C, H, W = mask_features.shape

        # inst_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feats)

        outputs_mask = torch.bmm(mask_embed, mask_features.view(B, C, H * W))
        outputs_mask = inst_masks.view(B, N, H, W)
        
        outputs_boxes = pred_bboxes.sigmoid()

        return pred_logits, outputs_mask, outputs_boxes


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, out_boxes=None):
        return [
            {"pred_logits": a, "pred_instance_masks": b, "pred_bboxes": c}
            for a, b, c in zip(outputs_class, outputs_seg_masks, out_boxes)
        ]
    

    def process_outputs(self, results, ori_shape):
        logits = results["logits"]
        scores = results["objectness_scores"]
        mask_embed = results["mask_embed"]
        bboxes = results["bboxes"]['instance_bboxes']
        mask_feats = results["mask_feats"]
        iams = results.get('iams')
        inst_pixel_attn = results.get('inst_pixel_attn')
        mask_pixel_attn = results.get('mask_pixel_attn')
        query_sa_attn = results.get('query_sa_attn')
        aux_outputs = results.get('aux_outputs')
        
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
            }, 
            'attn': {
                "inst_pixel_attn": inst_pixel_attn,
                "mask_pixel_attn": mask_pixel_attn, 
                "query_sa_attn": query_sa_attn
            },
            # 'aux_outputs': aux_outputs
        }
    
        return output

