import torch 
from torch import nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

import sys
sys.path.append('.')

from .instance_head import InstanceBranch
from .iam import (IAM)
from ..common import _make_stack_3x3_convs
from models.seg.nn.blocks import MLP, FFNLayer, CrossAttentionLayer, SelfAttentionLayer, PositionEmbeddingSine
from utils.registry import HEADS

    
    
# @HEADS.register(name="InstanceHead-v2.0-overlaps")
@HEADS.register(name="InstanceHead-v2.0-overlaps-attn")
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

        self.num_layers = 1
        
        # iam.
        self.inst_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        self.overlap_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        self.visible_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)

        hidden_dim = self.dim * self.num_groups
        dim_feedforward = 2048

        # positional encoding.
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.c_feats_proj = nn.Linear(hidden_dim*2, hidden_dim)

        # ca.
        self.transformer_overlap_cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=hidden_dim,
                nhead=8,
                dropout=0.0,
                normalize_before=False,
            )
            for _ in range(self.num_layers)
        ])
        
        self.transformer_visible_cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=hidden_dim,
                nhead=8,
                dropout=0.0,
                normalize_before=False,
            )
            for _ in range(self.num_layers)
        ])

        self.transformer_instance_cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=hidden_dim,
                nhead=8,
                dropout=0.0,
                normalize_before=False,
            )
            for _ in range(self.num_layers)
        ])
        
        # sa.
        self.transformer_overlaps_self_attention_layers = nn.ModuleList([
            SelfAttentionLayer(
                d_model=hidden_dim,
                nhead=8,
                dropout=0.0,
                normalize_before=False,
            )
            for _ in range(self.num_layers)
        ])

        self.transformer_visible_self_attention_layers = nn.ModuleList([
            SelfAttentionLayer(
                d_model=hidden_dim,
                nhead=8,
                dropout=0.0,
                normalize_before=False,
            )
            for _ in range(self.num_layers)
        ])

        self.transformer_instance_self_attention_layers = nn.ModuleList([
            SelfAttentionLayer(
                d_model=hidden_dim,
                nhead=8,
                dropout=0.0,
                normalize_before=False,
            )
            for _ in range(self.num_layers)
        ])

        # ffn.
        self.transformer_instance_ffn_layers = nn.ModuleList([
            FFNLayer(
                d_model=hidden_dim,
                dim_feedforward=dim_feedforward, 
                dropout=0.0, 
                normalize_before=False
            )
            for _ in range(self.num_layers)
        ])

        self.transformer_overlap_ffn_layers = nn.ModuleList([
            FFNLayer(
                d_model=hidden_dim,
                dim_feedforward=dim_feedforward, 
                dropout=0.0, 
                normalize_before=False
            )
            for _ in range(self.num_layers)
        ])

        self.transformer_visible_ffn_layers = nn.ModuleList([
            FFNLayer(
                d_model=hidden_dim,
                dim_feedforward=dim_feedforward, 
                dropout=0.0, 
                normalize_before=False
            )
            for _ in range(self.num_layers)
        ])

        self.decoder_norm = nn.LayerNorm(hidden_dim)
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
        print("v2.0-overlaps-attn")


    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, bias_value)
        init.normal_(self.inst_kernel.weight, std=0.01)
        init.constant_(self.inst_kernel.bias, 0.0)
        init.normal_(self.overlap_kernel.weight, std=0.01)
        init.constant_(self.overlap_kernel.bias, 0.0)
        init.normal_(self.visible_kernel.weight, std=0.01)
        init.constant_(self.visible_kernel.bias, 0.0)

    
    def forward_one_layer(self, inst_features, overlap_features, visible_features, query_embed, pos, i):
        # ca.
        overlap_features = self.transformer_overlap_cross_attention_layers[i](
            overlap_features, inst_features, 
            memory_mask=None, 
            memory_key_padding_mask=None, 
            pos=pos, query_pos=query_embed
            )
        
        visible_features = self.transformer_visible_cross_attention_layers[i](
            visible_features, inst_features, 
            memory_mask=None, 
            memory_key_padding_mask=None, 
            pos=pos, query_pos=query_embed
            )
        
        # not sure about this...
        memory = overlap_features + visible_features
        
        inst_features = self.transformer_instance_cross_attention_layers[i](
            inst_features, memory, 
            memory_mask=None, 
            memory_key_padding_mask=None, 
            pos=pos, query_pos=query_embed
            )
        
        # sa.
        overlap_features = self.transformer_overlaps_self_attention_layers[i](
            overlap_features, tgt_mask=None, 
            tgt_key_padding_mask=None, 
            query_pos=query_embed
            )
        
        visible_features = self.transformer_visible_self_attention_layers[i](
            visible_features, tgt_mask=None, 
            tgt_key_padding_mask=None, 
            query_pos=query_embed
            )
            
        inst_features = self.transformer_instance_self_attention_layers[i](
            inst_features, tgt_mask=None, 
            tgt_key_padding_mask=None, 
            query_pos=query_embed
            )
        
        # ffn.
        overlap_features = self.transformer_overlap_ffn_layers[i](overlap_features)
        visible_features = self.transformer_visible_ffn_layers[i](visible_features)
        inst_features = self.transformer_instance_ffn_layers[i](inst_features)
        
        return inst_features, overlap_features, visible_features


    def forward(self, features, prev_inst_features=None):
        inst_iam = self.inst_iam(features)
        overlap_iam = self.overlap_iam(features)
        visible_iam = self.visible_iam(features)

        B, N, H, W = inst_iam.shape
        C = features.size(1)
        
        if self.activation == "softmax":
            inst_iam_prob = F.softmax(inst_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
            overlap_iam_prob = F.softmax(overlap_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
            visible_iam_prob = F.softmax(visible_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
        else:
            raise NotImplementedError(f"No activation {self.activation} found!")
        
        inst_features = torch.bmm(inst_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        inst_features = inst_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        overlap_features = torch.bmm(overlap_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        overlap_features = overlap_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        visible_features = torch.bmm(visible_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        visible_features = visible_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
        
        inst_features = inst_features.transpose(0, 1)
        overlap_features = overlap_features.transpose(0, 1)
        visible_features = visible_features.transpose(0, 1)


        pos = self.pe_layer(features, None)
        print(pos.shape)
        pos = torch.bmm(inst_iam_prob, pos.view(B, C, -1).permute(0, 2, 1))
        pos = pos.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        pos = pos.transpose(0, 1)

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

        for i in range(self.num_layers):
            inst_features, overlap_features, visible_features = self.forward_one_layer(
                inst_features, overlap_features, visible_features, query_embed, pos, i
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




# @HEADS.register(name="InstanceHead-v2.0.1-overlaps-attn")
# class InstanceHead(nn.Module):
#     def __init__(self, 
#                  in_channels: int = 256, 
#                  num_convs: int = 4, 
#                  num_classes: int = 80, 
#                  kernel_dim: int = 256, 
#                  num_masks: int = 100, 
#                  num_groups: int = 1,
#                  activation: str = "softmax"):
#         super().__init__()
#         self.dim = in_channels
#         self.num_convs = num_convs
#         self.num_masks = num_masks
#         self.kernel_dim = kernel_dim
#         self.num_groups = num_groups
#         self.num_classes = num_classes + 1
#         self.activation = activation
#         self.scale_factor = 1
        
#         # IAMs
#         self.inst_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
#         self.overlap_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
#         self.visible_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)

#         hidden_dim = self.dim * self.num_groups

#         # Positional encoding
#         N_steps = hidden_dim // 2
#         self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
#         self.transformer_overlap_cross_attention_layers = nn.ModuleList(
#             [CrossAttentionLayer(d_model=hidden_dim, nhead=8, dropout=0.0, normalize_before=False) for _ in range(4)]
#         )
        
#         self.transformer_visible_cross_attention_layers = nn.ModuleList(
#             [CrossAttentionLayer(d_model=hidden_dim, nhead=8, dropout=0.0, normalize_before=False) for _ in range(4)]
#         )
        
#         self.transformer_self_attention_layer = nn.ModuleList(
#             [SelfAttentionLayer(d_model=hidden_dim, nhead=8, dropout=0.0, normalize_before=False) for _ in range(4)]
#         )

#         self.ffn_i = FFNLayer(hidden_dim)
#         self.ffn_o = FFNLayer(hidden_dim)
#         self.ffn_v = FFNLayer(hidden_dim)

#         self.decoder_norm = nn.LayerNorm(hidden_dim)

#         self.query_embed = nn.Embedding(self.num_masks, hidden_dim)
#         self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

#         # Outputs
#         self.cls_score = nn.Linear(hidden_dim, self.num_classes)
#         self.inst_kernel = MLP(hidden_dim, hidden_dim, self.kernel_dim, 3)
#         self.overlap_kernel = MLP(hidden_dim, hidden_dim, self.kernel_dim, 3)
#         self.visible_kernel = MLP(hidden_dim, hidden_dim, self.kernel_dim, 3)
#         self.objectness = nn.Linear(hidden_dim, 1)
#         self.bbox_pred = nn.Linear(hidden_dim, 4)
        
#         self.prior_prob = 0.01
#         self._init_weights()
        
#         self.softmax_bias = nn.Parameter(torch.ones([1, ]))
#         print("v2.0-overlaps")

#     def _init_weights(self):
#         bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
#         init.normal_(self.cls_score.weight, std=0.01)
#         init.constant_(self.cls_score.bias, bias_value)

#     def forward(self, features, prev_inst_features=None):
#         # Extract IAMs
#         inst_iam = self.inst_iam(features)  # (B, N, H, W)
#         overlap_iam = self.overlap_iam(features)  # (B, N, H, W)
#         visible_iam = self.visible_iam(features)  # (B, N, H, W)

#         B, N, H, W = inst_iam.shape
#         C = features.size(1)
        
#         # IAM Probabilities
#         if self.activation == "softmax":
#             inst_iam_prob = F.softmax(inst_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
#             overlap_iam_prob = F.softmax(overlap_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
#             visible_iam_prob = F.softmax(visible_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
#         else:
#             raise NotImplementedError(f"No activation {self.activation} found!")
        
#         # Instance Features
#         inst_features = torch.bmm(inst_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
#         inst_features = inst_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)  # (B, N//num_groups, hidden_dim)

#         # Overlap Features
#         overlap_features = torch.bmm(overlap_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
#         overlap_features = overlap_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)  # (B, N//num_groups, hidden_dim)

#         # Visible Features
#         visible_features = torch.bmm(visible_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
#         visible_features = visible_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)  # (B, N//num_groups, hidden_dim)
        
#         inst_features = inst_features.transpose(0, 1)  # (N//num_groups, B, hidden_dim)
#         overlap_features = overlap_features.transpose(0, 1)  # (N//num_groups, B, hidden_dim)
#         visible_features = visible_features.transpose(0, 1)  # (N//num_groups, B, hidden_dim)

#         # Positional Encoding
#         pos = self.pe_layer(features, None)  # (B, C, H, W)
#         pos = torch.bmm(inst_iam_prob, pos.view(B, C, -1).permute(0, 2, 1))
#         pos = pos.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
#         pos = pos.transpose(0, 1)  # (N//num_groups, B, hidden_dim)

#         query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (num_masks, B, hidden_dim)

#         # Cross-Attention for Overlap Features
#         for layer in self.transformer_overlap_cross_attention_layers:
#             overlap_features = layer(overlap_features, inst_features, memory_mask=None, memory_key_padding_mask=None, pos=pos, query_pos=query_embed)
        
#         # Cross-Attention for Visible Features
#         for layer in self.transformer_visible_cross_attention_layers:
#             visible_features = layer(visible_features, inst_features, memory_mask=None, memory_key_padding_mask=None, pos=pos, query_pos=query_embed)
        
#         # Combine Features
#         inst_features_o = overlap_features
#         inst_features_v = visible_features

#         # Feature Fusion
#         inst_features = torch.cat([inst_features_o, inst_features_v], dim=-1)  # (N//num_groups, B, hidden_dim * 2)
#         inst_features = self.proj(inst_features)  # (N//num_groups, B, hidden_dim)

#         # Self-Attention and FFN for Combined Features
#         for layer in self.transformer_self_attention_layer:
#             inst_features = layer(inst_features, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed)
        
#         overlap_features = self.ffn_o(overlap_features)
#         visible_features = self.ffn_v(visible_features)
#         inst_features = self.ffn_i(inst_features)

#         overlap_features = self.decoder_norm(overlap_features)
#         visible_features = self.decoder_norm(visible_features)
#         inst_features = self.decoder_norm(inst_features)

#         inst_features = inst_features.transpose(0, 1)
#         overlap_features = overlap_features.transpose(0, 1)
#         visible_features = visible_features.transpose(0, 1)

#         # Predictions
#         pred_logits = self.cls_score(refined_inst_features)
        
#         pred_inst_kernel = self.inst_kernel(refined_inst_features)
#         pred_overlap_kernel = self.overlap_kernel(inst_features_o)
#         pred_visible_kernel = self.visible_kernel(inst_features_v)

#         pred_scores = self.objectness(refined_inst_features)
#         pred_bboxes = self.bbox_pred(refined_inst_features)

#         results = {
#             'logits': pred_logits,
#             'objectness_scores': pred_scores,
#             'kernels': {
#                 'instance_kernel': pred_inst_kernel,
#                 'overlap_kernel': pred_overlap_kernel,
#                 'visible_kernel': pred_visible_kernel
#                 },
#             'bboxes': {
#                 'instance_bboxes': pred_bboxes
#                 },
#             'iams': {
#                 'instance_iams': inst_iam,
#                 'overlap_iams': overlap_iam,
#                 'visible_iams': visible_iam
#                 },
#             'inst_feats': {
#                 'instance_feats': refined_inst_features
#                 }
#         }

#         return results




@HEADS.register(name="InstanceHead-v2.1-overlaps")
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
        
        # iam prediction, a simple conv
        self.full_mask_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        self.overlap_mask_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        self.visible_mask_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        
        hidden_dim = self.dim * self.num_groups
        self.fc_f = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_fuse = nn.Linear(hidden_dim*2, hidden_dim)

        self.mixer = MLP(hidden_dim*3, hidden_dim*6, hidden_dim, 2)

        # Outputs
        self.cls_score = nn.Linear(hidden_dim, self.num_classes)
        self.full_mask_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.overlap_mask_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.visible_mask_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.objectness = nn.Linear(hidden_dim, 1)
        self.bbox_pred = nn.Linear(hidden_dim, 4)
        
        self.prior_prob = 0.01
        self._init_weights()
        
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))
        print("v2.1-overlaps")


    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, bias_value)

        init.normal_(self.full_mask_kernel.weight, std=0.01)
        init.constant_(self.full_mask_kernel.bias, 0.0)

        init.normal_(self.overlap_mask_kernel.weight, std=0.01)
        init.constant_(self.overlap_mask_kernel.bias, 0.0)

        init.normal_(self.visible_mask_kernel.weight, std=0.01)
        init.constant_(self.visible_mask_kernel.bias, 0.0)

        c2_xavier_fill(self.fc_f)
        c2_xavier_fill(self.fc_o)
        c2_xavier_fill(self.fc_v)
        c2_xavier_fill(self.fc_fuse)


    def forward(self, features, prev_inst_features=None):
        full_mask_iam = self.full_mask_iam(features)
        overlap_mask_iam = self.overlap_mask_iam(features)
        visible_mask_iam = self.visible_mask_iam(features)

        B, N, H, W = full_mask_iam.shape
        C = features.size(1)
        
        if self.activation == "softmax":
            full_mask_iam_prob = F.softmax(full_mask_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
            overlap_mask_iam_prob = F.softmax(overlap_mask_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
            visible_mask_iam_prob = F.softmax(visible_mask_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
        else:
            raise NotImplementedError(f"No activation {self.activation} found!")
        
        full_mask_inst_features = torch.bmm(full_mask_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        full_mask_inst_features = full_mask_inst_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        overlap_mask_inst_features = torch.bmm(overlap_mask_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        overlap_mask_inst_features = overlap_mask_inst_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        visible_mask_inst_features = torch.bmm(visible_mask_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        visible_mask_inst_features = visible_mask_inst_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)


        full_mask_inst_features = F.relu_(self.fc_f(full_mask_inst_features))
        overlap_mask_inst_features = F.relu_(self.fc_o(overlap_mask_inst_features))
        visible_mask_inst_features = F.relu_(self.fc_v(visible_mask_inst_features))

        # inst feats refining.
        inst_features = torch.cat([full_mask_inst_features, 
                                   overlap_mask_inst_features, 
                                   visible_mask_inst_features], dim=-1)

        full_mask_inst_features = self.mixer(inst_features)


        # predictions.
        pred_logits = self.cls_score(full_mask_inst_features)
        
        pred_full_mask_kernel = self.full_mask_kernel(full_mask_inst_features)
        pred_overlap_mask_kernel = self.overlap_mask_kernel(overlap_mask_inst_features)
        pred_visible_mask_kernel = self.visible_mask_kernel(visible_mask_inst_features)

        pred_scores = self.objectness(full_mask_inst_features)
        pred_bboxes = self.bbox_pred(full_mask_inst_features)

        results = {
            'logits': pred_logits,
            'mask_kernel': pred_full_mask_kernel,
            'overlap_mask_kernel': pred_overlap_mask_kernel,
            'visible_mask_kernel': pred_visible_mask_kernel,
            'objectness_scores': pred_scores,
            'bboxes': pred_bboxes,
            'iam': full_mask_iam,
            'inst_feats': full_mask_inst_features
        }

        return results




@HEADS.register(name="InstanceHead-v2.2-overlaps")
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


        # branches.
        self.instance_branch = _make_stack_3x3_convs(
            in_channels=self.dim, 
            out_channels=self.dim, 
            num_convs=3
            )
        
        self.overlap_branch = _make_stack_3x3_convs(
            in_channels=self.dim * 2, 
            out_channels=self.dim, 
            num_convs=3
            )
        
        self.visible_branch = _make_stack_3x3_convs(
            in_channels=self.dim * 2, 
            out_channels=self.dim, 
            num_convs=3
            )

        self.r_instance_branch = _make_stack_3x3_convs(
            in_channels=self.dim * 3, 
            out_channels=self.dim, 
            num_convs=3
            )
        
        # iam.
        self.inst_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        self.overlap_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        self.visible_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        
        hidden_dim = self.dim * self.num_groups
        self.fc_f = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_fuse = nn.Linear(hidden_dim*2, hidden_dim)

        # outputs.
        self.cls_score = nn.Linear(hidden_dim, self.num_classes)
        self.inst_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.overlap_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.visible_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.objectness = nn.Linear(hidden_dim, 1)
        self.bbox_pred = nn.Linear(hidden_dim, 4)
        
        self.prior_prob = 0.01
        self._init_weights()
        
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))
        print("v2.2-overlaps")

    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, bias_value)

        init.normal_(self.inst_kernel.weight, std=0.01)
        init.constant_(self.inst_kernel.bias, 0.0)

        init.normal_(self.overlap_kernel.weight, std=0.01)
        init.constant_(self.overlap_kernel.bias, 0.0)

        init.normal_(self.visible_kernel.weight, std=0.01)
        init.constant_(self.visible_kernel.bias, 0.0)

        c2_xavier_fill(self.fc_f)
        c2_xavier_fill(self.fc_o)
        c2_xavier_fill(self.fc_v)
        c2_xavier_fill(self.fc_fuse)

        for m in self.instance_branch.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

        for m in self.overlap_branch.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

        for m in self.visible_branch.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

        for m in self.r_instance_branch.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
                

    def forward(self, features, prev_inst_features=None):
        inst_features = self.instance_branch(features)
        f_i = torch.cat([inst_features, features], dim=1)
        overlap_features = self.overlap_branch(f_i)
        visible_features = self.visible_branch(f_i)

        f_r = torch.cat([inst_features, overlap_features, visible_features], dim=1)
        inst_features = self.r_instance_branch(f_r)


        inst_iam = self.inst_iam(inst_features)
        overlap_iam = self.overlap_iam(overlap_features)
        visible_iam = self.visible_iam(visible_features)

        B, N, H, W = inst_iam.shape
        C = inst_features.size(1)
        
        if self.activation == "softmax":
            inst_iam_prob = F.softmax(inst_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
            overlap_iam_prob = F.softmax(overlap_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
            visible_iam_prob = F.softmax(visible_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
        else:
            raise NotImplementedError(f"No activation {self.activation} found!")
        
        inst_features = torch.bmm(inst_iam_prob, inst_features.view(B, C, -1).permute(0, 2, 1))
        inst_features = inst_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        overlap_features = torch.bmm(overlap_iam_prob, overlap_features.view(B, C, -1).permute(0, 2, 1))
        overlap_features = overlap_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        visible_features = torch.bmm(visible_iam_prob, visible_features.view(B, C, -1).permute(0, 2, 1))
        visible_features = visible_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        inst_features = F.relu_(self.fc_f(inst_features))
        overlap_features = F.relu_(self.fc_o(overlap_features))
        visible_features = F.relu_(self.fc_v(visible_features))


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
    




@HEADS.register(name="InstanceHead-v2.3-overlaps")
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


        # branches.
        self.instance_branch = _make_stack_3x3_convs(
            in_channels=self.dim, 
            out_channels=self.dim, 
            num_convs=3
            )
        
        self.overlap_branch = _make_stack_3x3_convs(
            in_channels=self.dim * 2, 
            out_channels=self.dim, 
            num_convs=3
            )
        
        self.visible_branch = _make_stack_3x3_convs(
            in_channels=self.dim * 2, 
            out_channels=self.dim, 
            num_convs=3
            )

        self.r_instance_branch = _make_stack_3x3_convs(
            in_channels=self.dim * 3, 
            out_channels=self.dim, 
            num_convs=3
            )
        
        # iam.
        self.inst_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        self.overlap_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        self.visible_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)
        
        hidden_dim = self.dim * self.num_groups
        self.fc_f = nn.Linear(hidden_dim, hidden_dim)
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        self.fc_fuse = nn.Linear(hidden_dim*2, hidden_dim)

        # outputs.
        self.cls_score = nn.Linear(hidden_dim, self.num_classes)
        self.inst_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.overlap_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.visible_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.objectness = nn.Linear(hidden_dim, 1)
        self.bbox_pred = nn.Linear(hidden_dim, 4)
        
        self.prior_prob = 0.01
        self._init_weights()
        
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))
        print("v2.3-overlaps")

    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, bias_value)

        init.normal_(self.inst_kernel.weight, std=0.01)
        init.constant_(self.inst_kernel.bias, 0.0)

        init.normal_(self.overlap_kernel.weight, std=0.01)
        init.constant_(self.overlap_kernel.bias, 0.0)

        init.normal_(self.visible_kernel.weight, std=0.01)
        init.constant_(self.visible_kernel.bias, 0.0)

        c2_xavier_fill(self.fc_f)
        c2_xavier_fill(self.fc_o)
        c2_xavier_fill(self.fc_v)
        c2_xavier_fill(self.fc_fuse)

        for m in self.instance_branch.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

        for m in self.overlap_branch.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

        for m in self.visible_branch.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

        for m in self.r_instance_branch.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
                

    def forward(self, features, prev_inst_features=None):
        inst_features = self.instance_branch(features)
        f_i = torch.cat([inst_features, features], dim=1)
        overlap_features = self.overlap_branch(f_i)
        visible_features = self.visible_branch(f_i)

        f_r = torch.cat([inst_features, overlap_features, visible_features], dim=1)
        inst_features = self.r_instance_branch(f_r)


        inst_iam = self.inst_iam(inst_features)
        overlap_iam = self.overlap_iam(overlap_features)
        visible_iam = self.visible_iam(visible_features)

        B, N, H, W = inst_iam.shape
        C = inst_features.size(1)
        
        if self.activation == "softmax":
            inst_iam_prob = F.softmax(inst_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
            overlap_iam_prob = F.softmax(overlap_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
            visible_iam_prob = F.softmax(visible_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
        else:
            raise NotImplementedError(f"No activation {self.activation} found!")
        
        inst_features = torch.bmm(inst_iam_prob, inst_features.view(B, C, -1).permute(0, 2, 1))
        inst_features = inst_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        overlap_features = torch.bmm(overlap_iam_prob, overlap_features.view(B, C, -1).permute(0, 2, 1))
        overlap_features = overlap_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        visible_features = torch.bmm(visible_iam_prob, visible_features.view(B, C, -1).permute(0, 2, 1))
        visible_features = visible_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)

        inst_features = F.relu_(self.fc_f(inst_features))
        overlap_features = F.relu_(self.fc_o(overlap_features))
        visible_features = F.relu_(self.fc_v(visible_features))


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