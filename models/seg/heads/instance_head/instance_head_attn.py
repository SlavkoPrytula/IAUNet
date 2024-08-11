import torch 
from torch import nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F

import sys
sys.path.append('.')

from .iam import (IAM)
from models.seg.nn.blocks import MLP, FFNLayer
from models.seg.nn.blocks import CrossAttentionLayer, SelfAttentionLayer, PositionEmbeddingSine
from utils.registry import HEADS

    
    

@HEADS.register(name="InstanceHead-v2.0-attn")
class InstanceHead(nn.Module):
    def __init__(self, 
                 in_channels: int = 256, 
                 num_convs: int = 4, 
                 num_classes: int = 80, 
                 kernel_dim: int = 256, 
                 num_masks: int = 100, 
                 num_groups: int = 1,
                 activation: str = "softmax", 
                 num_layers: int = 1,
                 nhead: int = 8, 
                 dim_feedforward: int = 2048,
                 dropout: float = 0.0,
                 normalize_before: bool = False):
        super().__init__()
        self.dim = in_channels
        self.num_convs = num_convs
        self.num_masks = num_masks
        self.kernel_dim = kernel_dim
        self.num_groups = num_groups
        self.num_classes = num_classes + 1
        self.activation = activation
        self.scale_factor = 1

        self.num_layers = num_layers
        hidden_dim = self.dim * self.num_groups
        
        # iam.
        self.inst_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)

        # positional encoding.
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # ca.
        self.transformer_instance_cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dropout=dropout,
                normalize_before=normalize_before,
            )
            for _ in range(self.num_layers)
        ])
        
        # sa.
        self.transformer_instance_self_attention_layers = nn.ModuleList([
            SelfAttentionLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dropout=dropout,
                normalize_before=normalize_before,
            )
            for _ in range(self.num_layers)
        ])

        # ffn.
        self.transformer_instance_ffn_layers = nn.ModuleList([
            FFNLayer(
                d_model=hidden_dim,
                dim_feedforward=dim_feedforward, 
                dropout=dropout, 
                normalize_before=normalize_before
            )
            for _ in range(self.num_layers)
        ])

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.query_embed = nn.Embedding(self.num_masks, hidden_dim)
        self.pos_embed = nn.Embedding(self.num_masks, hidden_dim)

        # Outputs
        self.cls_score = nn.Linear(hidden_dim, self.num_classes)
        self.inst_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.objectness = nn.Linear(hidden_dim, 1)
        self.bbox_pred = nn.Linear(hidden_dim, 4)
        
        self.prior_prob = 0.01
        self._init_weights()
        
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))
        print("v2.0-instance-attn")


    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, bias_value)
        init.normal_(self.inst_kernel.weight, std=0.01)
        init.constant_(self.inst_kernel.bias, 0.0)

    
    def forward_one_layer(self, inst_features, pixel_features, query_embed, pos, i):
        # ca.
        inst_features = self.transformer_instance_cross_attention_layers[i](
            inst_features, pixel_features, 
            memory_mask=None, 
            memory_key_padding_mask=None, 
            pos=pos, query_pos=query_embed
            )
        
        # sa.
        inst_features = self.transformer_instance_self_attention_layers[i](
            inst_features, tgt_mask=None, 
            tgt_key_padding_mask=None, 
            query_pos=query_embed
            )
        
        # ffn.
        inst_features = self.transformer_instance_ffn_layers[i](inst_features)
        
        return inst_features


    def forward(self, features, prev_inst_features=None):
        inst_iam = self.inst_iam(features)

        B, N, H, W = inst_iam.shape
        C = features.size(1)
        
        if self.activation == "softmax":
            inst_iam_prob = F.softmax(inst_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
        else:
            raise NotImplementedError(f"No activation {self.activation} found!")
        
        inst_features = torch.bmm(inst_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        inst_features = inst_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
        inst_features = inst_features.transpose(0, 1)
        print(features.shape)
        pixel_features = features.flatten(2).permute(2, 0, 1)
        print(pixel_features.shape)
        raise

        # pos = self.pe_layer(features, None)
        # pos = torch.bmm(inst_iam_prob, pos.view(B, C, -1).permute(0, 2, 1))
        # pos = pos.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        # pos = pos.transpose(0, 1)
        pos = self.pos_embed.weight.unsqueeze(1).repeat(1, B, 1)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

        for i in range(self.num_layers):
            inst_features = self.forward_one_layer(
                inst_features, pixel_features, query_embed, pos, i
            )

        overlap_features = self.decoder_norm(overlap_features)
        visible_features = self.decoder_norm(visible_features)
        inst_features = self.decoder_norm(inst_features)

        inst_features = inst_features.transpose(0, 1)
        overlap_features = overlap_features.transpose(0, 1)
        visible_features = visible_features.transpose(0, 1)


        # predictions.
        pred_logits = self.cls_score(inst_features)
        
        pred_inst_kernel = self.inst_kernel(inst_features)
        pred_overlap_kernel = self.overlap_kernel(overlap_features)
        pred_visible_kernel = self.visible_kernel(visible_features)

        pred_scores = self.objectness(inst_features)
        pred_bboxes = self.bbox_pred(inst_features)


        results = {
            'logits': pred_logits,
            'objectness_scores': pred_scores,
            'kernels': {
                'instance_kernel': pred_inst_kernel,
                'overlap_kernel': pred_overlap_kernel,
                'visible_kernel': pred_visible_kernel
                },
            'bboxes': {
                'instance_bboxes': pred_bboxes
                },
            'iams': {
                'instance_iams': inst_iam,
                'overlap_iams': overlap_iam,
                'visible_iams': visible_iam
                },
            'inst_feats': {
                'instance_feats': inst_features
                }
        }

        return results
    



@HEADS.register(name="InstanceHead-v2.0.1-overlaps-attn")
class InstanceHead(nn.Module):
    def __init__(self, 
                 in_channels: int = 256, 
                 num_convs: int = 4, 
                 num_classes: int = 80, 
                 kernel_dim: int = 256, 
                 num_masks: int = 100, 
                 num_groups: int = 1,
                 activation: str = "softmax"):
        super().__init__()
        self.dim = in_channels
        self.num_convs = num_convs
        self.num_masks = num_masks
        self.kernel_dim = kernel_dim
        self.num_groups = num_groups
        self.num_classes = num_classes + 1
        self.activation = activation
        self.scale_factor = 1
        
        # iam.
        self.inst_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)

        hidden_dim = self.dim * self.num_groups

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        n_layers = 1
        self.transformer_overlap_cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=hidden_dim,
                nhead=8,
                dropout=0.0,
                normalize_before=False,
            )
            for _ in range(n_layers)
        ])
        
        self.transformer_visible_cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=hidden_dim,
                nhead=8,
                dropout=0.0,
                normalize_before=False,
            )
            for _ in range(n_layers)
        ])
        
        self.transformer_overlaps_self_attention_layers = nn.ModuleList([
            SelfAttentionLayer(
                d_model=hidden_dim,
                nhead=8,
                dropout=0.0,
                normalize_before=False,
            )
            for _ in range(n_layers)
        ])

        self.transformer_visible_self_attention_layers = nn.ModuleList([
            SelfAttentionLayer(
                d_model=hidden_dim,
                nhead=8,
                dropout=0.0,
                normalize_before=False,
            )
            for _ in range(n_layers)
        ])

        self.transformer_instance_self_attention_layers = nn.ModuleList([
            SelfAttentionLayer(
                d_model=hidden_dim,
                nhead=8,
                dropout=0.0,
                normalize_before=False,
            )
            for _ in range(n_layers)
        ])

        self.ffn_i = FFNLayer(hidden_dim)
        self.ffn_o = FFNLayer(hidden_dim)
        self.ffn_v = FFNLayer(hidden_dim)

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        # learnable query features
        self.overlap_features = nn.Embedding(self.num_masks, hidden_dim)
        self.visible_features = nn.Embedding(self.num_masks, hidden_dim)

        self.query_embed = nn.Embedding(self.num_masks, hidden_dim)

        # Outputs
        self.cls_score = nn.Linear(hidden_dim, self.num_classes)
        self.inst_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.overlap_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.visible_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.objectness = nn.Linear(hidden_dim, 1)
        self.bbox_pred = nn.Linear(hidden_dim, 4)
        
        self.prior_prob = 0.01
        self._init_weights()
        
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))
        print("v2.0.1-overlaps-attn")


    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, bias_value)


    def forward(self, features, prev_inst_features=None):
        inst_iam = self.inst_iam(features)

        B, N, H, W = inst_iam.shape
        C = features.size(1)
        
        if self.activation == "softmax":
            inst_iam_prob = F.softmax(inst_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
        else:
            raise NotImplementedError(f"No activation {self.activation} found!")
        
        inst_features = torch.bmm(inst_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        inst_features = inst_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        inst_features = inst_features.transpose(0, 1)
        
        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

        overlap_features = self.overlap_features.weight.unsqueeze(1).repeat(1, B, 1)
        visible_features = self.visible_features.weight.unsqueeze(1).repeat(1, B, 1)

        # pe.
        pos = self.pe_layer(features, None)
        pos = torch.bmm(inst_iam_prob, pos.view(B, C, -1).permute(0, 2, 1))
        pos = pos.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        pos = pos.transpose(0, 1)


        # ca.
        for layer in self.transformer_overlap_cross_attention_layers:
            overlap_features = layer(
                overlap_features, inst_features, 
                memory_mask=None, 
                memory_key_padding_mask=None, 
                pos=pos, query_pos=query_embed
                )
        
        for layer in self.transformer_visible_cross_attention_layers:
            visible_features = layer(
                visible_features, inst_features, 
                memory_mask=None, 
                memory_key_padding_mask=None, 
                pos=pos, query_pos=query_embed
                )
        
        # sa.
        for layer in self.transformer_overlaps_self_attention_layers:
            overlap_features = layer(
                overlap_features, tgt_mask=None, 
                tgt_key_padding_mask=None, 
                query_pos=query_embed
                )
        
        for layer in self.transformer_visible_self_attention_layers:
            visible_features = layer(
                visible_features, tgt_mask=None, 
                tgt_key_padding_mask=None, 
                query_pos=query_embed
                )
            
        for layer in self.transformer_instance_self_attention_layers:
            inst_features = layer(
                inst_features, tgt_mask=None, 
                tgt_key_padding_mask=None, 
                query_pos=query_embed
                )
        
        overlap_features = self.ffn_o(overlap_features)
        visible_features = self.ffn_v(visible_features)
        inst_features = self.ffn_i(inst_features)

        overlap_features = self.decoder_norm(overlap_features)
        visible_features = self.decoder_norm(visible_features)
        inst_features = self.decoder_norm(inst_features)

        inst_features = inst_features.transpose(0, 1)
        overlap_features = overlap_features.transpose(0, 1)
        visible_features = visible_features.transpose(0, 1)


        # predictions.
        pred_logits = self.cls_score(inst_features)
        
        pred_inst_kernel = self.inst_kernel(inst_features)
        pred_overlap_kernel = self.overlap_kernel(overlap_features)
        pred_visible_kernel = self.visible_kernel(visible_features)

        pred_scores = self.objectness(inst_features)
        pred_bboxes = self.bbox_pred(inst_features)


        results = {
            'logits': pred_logits,
            'objectness_scores': pred_scores,
            'kernels': {
                'instance_kernel': pred_inst_kernel,
                'overlap_kernel': pred_overlap_kernel,
                'visible_kernel': pred_visible_kernel
                },
            'bboxes': {
                'instance_bboxes': pred_bboxes
                },
            'iams': {
                'instance_iams': inst_iam,
                },
            'inst_feats': {
                'instance_feats': inst_features
                }
        }

        return results

