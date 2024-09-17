import torch 
from torch import nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
from fvcore.nn.weight_init import c2_xavier_fill

import sys
sys.path.append('.')

from .iam import (IAM, DeepIAM)
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
                 dim_feedforward: int = 2048,
                 nhead: int = 8, 
                 normalize_before: bool = False,
                 dropout: float = 0.0):
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

        # Outputs
        # self.cls_score = nn.Linear(hidden_dim, self.num_classes)
        # self.inst_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.cls_score = MLP(hidden_dim, hidden_dim, self.num_classes, 3)
        self.inst_kernel = MLP(hidden_dim, hidden_dim, self.kernel_dim, 3)
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
        # inst_features = inst_features.reshape(B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
        inst_features = inst_features.transpose(0, 1)
        pixel_features = features.flatten(2).permute(2, 0, 1)

        pos = self.pe_layer(features, None)
        pos = pos.flatten(2).permute(2, 0, 1)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

        for i in range(self.num_layers):
            inst_features = self.forward_one_layer(
                inst_features, pixel_features, query_embed, pos, i
            )

        inst_features = self.decoder_norm(inst_features)
        inst_features = inst_features.transpose(0, 1)

        # predictions.
        pred_logits = self.cls_score(inst_features)
        pred_inst_kernel = self.inst_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        pred_bboxes = self.bbox_pred(inst_features)


        results = {
            'logits': pred_logits,
            'objectness_scores': pred_scores,
            'kernels': {
                'instance_kernel': pred_inst_kernel,
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
    







@HEADS.register(name="InstanceHead-v2.1-attn")
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
                 dim_feedforward: int = 2048,
                 nhead: int = 8, 
                 normalize_before: bool = False,
                 dropout: float = 0.0):
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
        self.prev_query_embed = nn.Embedding(self.num_masks, hidden_dim)

        # Outputs
        self.cls_score = nn.Linear(hidden_dim, self.num_classes)
        self.inst_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.objectness = nn.Linear(hidden_dim, 1)
        self.bbox_pred = nn.Linear(hidden_dim, 4)
        
        self.prior_prob = 0.01
        self._init_weights()
        
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))
        print("v2.1-instance-attn")


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
        
        inst_features = inst_features.transpose(0, 1)
        pixel_features = features.flatten(2).permute(2, 0, 1)

        pos = self.pe_layer(features, None)
        pos = pos.flatten(2).permute(2, 0, 1)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

        if prev_inst_features is not None:
            prev_inst_features = prev_inst_features.transpose(0, 1)
            inst_features = torch.cat([inst_features, prev_inst_features], dim=0)

            prev_query_embed = self.prev_query_embed.weight.unsqueeze(1).repeat(1, B, 1)
            query_embed = torch.cat([query_embed, prev_query_embed], dim=0)

        for i in range(self.num_layers):
            inst_features = self.forward_one_layer(
                inst_features, pixel_features, query_embed, pos, i
            )

        inst_features = inst_features[:self.num_masks]
        inst_features = self.decoder_norm(inst_features)
        inst_features = inst_features.transpose(0, 1)

        # predictions.
        pred_logits = self.cls_score(inst_features)
        pred_inst_kernel = self.inst_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        pred_bboxes = self.bbox_pred(inst_features)


        results = {
            'logits': pred_logits,
            'objectness_scores': pred_scores,
            'kernels': {
                'instance_kernel': pred_inst_kernel,
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
    





@HEADS.register(name="InstanceHead-v2.2-two-way-attn")
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
                 dim_feedforward: int = 2048,
                 nhead: int = 8, 
                 normalize_before: bool = False,
                 dropout: float = 0.0):
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


        # ca - feats.
        self.transformer_feats_cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dropout=dropout,
                normalize_before=normalize_before,
            )
            for _ in range(self.num_layers)
        ])

        # ffn - feats.
        self.transformer_feats_ffn_layers = nn.ModuleList([
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
        self.prev_query_embed = nn.Embedding(self.num_masks, hidden_dim)

        # Outputs
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.cls_score = nn.Linear(hidden_dim, self.num_classes)
        self.inst_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        self.objectness = nn.Linear(hidden_dim, 1)
        self.bbox_pred = nn.Linear(hidden_dim, 4)
        
        self.prior_prob = 0.01
        self._init_weights()
        
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))
        print("v2.2-instance-two-way-attn")


    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, bias_value)
        init.normal_(self.inst_kernel.weight, std=0.01)
        init.constant_(self.inst_kernel.bias, 0.0)

        c2_xavier_fill(self.fc)

    
    def forward_one_layer(self, inst_features, pixel_features, query_embed, pos, i):
        # ca.
        pixel_features = self.transformer_feats_cross_attention_layers[i](
            pixel_features, inst_features, 
            memory_mask=None, 
            memory_key_padding_mask=None, 
            pos=query_embed, query_pos=pos
            )
        
        # ffn.
        pixel_features = self.transformer_feats_ffn_layers[i](pixel_features)

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
        
        return inst_features, pixel_features


    def forward(self, features, mask_feats, prev_inst_features=None, attn_mask=None):
        inst_iam = self.inst_iam(features)

        B, N, H, W = inst_iam.shape
        C = features.size(1)
        
        if self.activation == "softmax":
            inst_iam_prob = F.softmax(inst_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
        else:
            raise NotImplementedError(f"No activation {self.activation} found!")
        
        inst_features = torch.bmm(inst_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        
        inst_features = inst_features.transpose(0, 1)
        pixel_features = features.flatten(2).permute(2, 0, 1)

        pos = self.pe_layer(features, None)
        pos = pos.flatten(2).permute(2, 0, 1)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

        if prev_inst_features is not None:
            prev_inst_features = prev_inst_features.transpose(0, 1)
            inst_features = torch.cat([inst_features, prev_inst_features], dim=0)
            
            prev_query_embed = self.prev_query_embed.weight.unsqueeze(1).repeat(1, B, 1)
            query_embed = torch.cat([query_embed, prev_query_embed], dim=0)

        for i in range(self.num_layers):    
            inst_features, pixel_features = self.forward_one_layer(
                inst_features, pixel_features, query_embed, pos, i
            )

        inst_features = inst_features[:self.num_masks]
        inst_features = self.decoder_norm(inst_features)
        inst_features = inst_features.transpose(0, 1)

        inst_features = F.relu_(self.fc(inst_features))

        # predictions.
        pred_logits = self.cls_score(inst_features)
        pred_inst_kernel = self.inst_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        pred_bboxes = self.bbox_pred(inst_features)

        # (L, B, C) -> (B, C, L) -> (B, C, H, W)
        pixel_features = pixel_features.permute(1, 2, 0).view(B, C, H, W)

        results = {
            'logits': pred_logits,
            'objectness_scores': pred_scores,
            'kernels': {
                'instance_kernel': pred_inst_kernel,
                },
            'bboxes': {
                'instance_bboxes': pred_bboxes
                },
            'iams': {
                'instance_iams': inst_iam,
                },
            'inst_feats': {
                'instance_feats': inst_features
                },
            'pixel_feats': pixel_features
        }

        return results
    




@HEADS.register(name="InstanceHead-v2.2.1-dual-update")
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
                 dim_feedforward: int = 2048,
                 nhead: int = 8, 
                 normalize_before: bool = False,
                 dropout: float = 0.0):
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


        # ca - inst feats.
        self.transformer_inst_feats_cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dropout=dropout,
                normalize_before=normalize_before,
            )
            for _ in range(self.num_layers)
        ])

        # ffn - inst feats.
        self.transformer_inst_feats_ffn_layers = nn.ModuleList([
            FFNLayer(
                d_model=hidden_dim,
                dim_feedforward=dim_feedforward, 
                dropout=dropout, 
                normalize_before=normalize_before
            )
            for _ in range(self.num_layers)
        ])

        # ca - mask feats.
        self.transformer_mask_feats_cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dropout=dropout,
                normalize_before=normalize_before,
            )
            for _ in range(num_layers)
        ])
        
        # ffn - mask feats.
        self.transformer_mask_feats_ffn_layers = nn.ModuleList([
            FFNLayer(
                d_model=hidden_dim,
                dim_feedforward=dim_feedforward, 
                dropout=dropout, 
                normalize_before=normalize_before
            )
            for _ in range(num_layers)
        ])

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.query_embed = nn.Embedding(self.num_masks, hidden_dim)
        self.prev_query_embed = nn.Embedding(self.num_masks, hidden_dim)

        # Outputs
        self.cls_score = nn.Linear(hidden_dim, self.num_classes)
        self.inst_kernel = nn.Linear(hidden_dim, self.kernel_dim)
        # self.cls_score = MLP(hidden_dim, hidden_dim, self.num_classes, 3)
        # self.inst_kernel = MLP(hidden_dim, hidden_dim, self.kernel_dim, 3)
        self.objectness = nn.Linear(hidden_dim, 1)
        self.bbox_pred = nn.Linear(hidden_dim, 4)
        
        self.prior_prob = 0.01
        self._init_weights()
        
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))
        print("v2.2.1-instance-dual-feat-update")


    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        # init.normal_(self.cls_score.weight, std=0.01)
        # init.constant_(self.cls_score.bias, bias_value)
        # init.normal_(self.inst_kernel.weight, std=0.01)
        # init.constant_(self.inst_kernel.bias, 0.0)

    
    def forward_one_layer(self, inst_features, inst_pixel_features, mask_pixel_features, query_embed, pos, i):
        # [inst_pixel_features, mask_pixel_features] -> inst_features
        # --------------
        # inst feats ca.
        inst_pixel_features = self.transformer_inst_feats_cross_attention_layers[i](
            inst_pixel_features, inst_features, 
            memory_mask=None, 
            memory_key_padding_mask=None, 
            pos=query_embed, query_pos=pos
            )
        
        # inst feats ffn.
        inst_pixel_features = self.transformer_inst_feats_ffn_layers[i](inst_pixel_features)


        # --------------
        # mask feats ca.
        mask_pixel_features = self.transformer_mask_feats_cross_attention_layers[i](
            mask_pixel_features, inst_features,
            memory_mask=None,
            memory_key_padding_mask=None,
            pos=query_embed, query_pos=pos
        )

        # mask feats ffn.
        mask_pixel_features = self.transformer_mask_feats_ffn_layers[i](mask_pixel_features)


        # --------------
        # inst embed ca.
        inst_features = self.transformer_instance_cross_attention_layers[i](
            inst_features, inst_pixel_features, 
            memory_mask=None, 
            memory_key_padding_mask=None, 
            pos=pos, query_pos=query_embed
            )
        
        # inst embed sa.
        inst_features = self.transformer_instance_self_attention_layers[i](
            inst_features, tgt_mask=None, 
            tgt_key_padding_mask=None, 
            query_pos=query_embed
            )
        
        # inst embed ffn.
        inst_features = self.transformer_instance_ffn_layers[i](inst_features)
        

        return inst_features, inst_pixel_features, mask_pixel_features


    # def forward_one_layer(self, inst_features, inst_pixel_features, mask_pixel_features, query_embed, pos, i):
    #     # inst_pixel_features -> inst_features -> mask_pixel_features
    #     # --------------
    #     # inst feats ca.
    #     inst_pixel_features = self.transformer_inst_feats_cross_attention_layers[i](
    #         inst_pixel_features, inst_features, 
    #         memory_mask=None, 
    #         memory_key_padding_mask=None, 
    #         pos=query_embed, query_pos=pos
    #         )
        
    #     # inst feats ffn.
    #     inst_pixel_features = self.transformer_inst_feats_ffn_layers[i](inst_pixel_features)


    #     # --------------
    #     # inst embed ca.
    #     inst_features = self.transformer_instance_cross_attention_layers[i](
    #         inst_features, inst_pixel_features, 
    #         memory_mask=None, 
    #         memory_key_padding_mask=None, 
    #         pos=pos, query_pos=query_embed
    #         )
        
    #     # inst embed sa.
    #     inst_features = self.transformer_instance_self_attention_layers[i](
    #         inst_features, tgt_mask=None, 
    #         tgt_key_padding_mask=None, 
    #         query_pos=query_embed
    #         )
        
    #     # inst embed ffn.
    #     inst_features = self.transformer_instance_ffn_layers[i](inst_features)


    #     # --------------
    #     # mask feats ca.
    #     mask_pixel_features = self.transformer_mask_feats_cross_attention_layers[i](
    #         mask_pixel_features, inst_features,
    #         memory_mask=None,
    #         memory_key_padding_mask=None,
    #         pos=query_embed, query_pos=pos
    #     )

    #     # mask feats ffn.
    #     mask_pixel_features = self.transformer_mask_feats_ffn_layers[i](mask_pixel_features)
        
    #     return inst_features, inst_pixel_features, mask_pixel_features


    def forward(self, features, mask_feats, prev_inst_features=None):
        inst_iam = self.inst_iam(features)

        B, N, H, W = inst_iam.shape
        C = features.size(1)
        
        if self.activation == "softmax":
            inst_iam_prob = F.softmax(inst_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
        else:
            raise NotImplementedError(f"No activation {self.activation} found!")
        
        inst_features = torch.bmm(inst_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        
        inst_features = inst_features.transpose(0, 1)
        inst_pixel_features = features.flatten(2).permute(2, 0, 1)
        mask_pixel_features = mask_feats.flatten(2).permute(2, 0, 1)

        pos = self.pe_layer(features, None)
        pos = pos.flatten(2).permute(2, 0, 1)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

        if prev_inst_features is not None:
            prev_inst_features = prev_inst_features.transpose(0, 1)
            inst_features = torch.cat([inst_features, prev_inst_features], dim=0)
            
            prev_query_embed = self.prev_query_embed.weight.unsqueeze(1).repeat(1, B, 1)
            query_embed = torch.cat([query_embed, prev_query_embed], dim=0)

        for i in range(self.num_layers):    
            inst_features, inst_pixel_features, mask_pixel_features = self.forward_one_layer(
                inst_features, inst_pixel_features, mask_pixel_features, query_embed, pos, i
            )

        inst_features = inst_features[:self.num_masks]
        inst_features = self.decoder_norm(inst_features)
        inst_features = inst_features.transpose(0, 1)

        # predictions.
        pred_logits = self.cls_score(inst_features)
        pred_inst_features = self.inst_kernel(inst_features)
        pred_scores = self.objectness(inst_features)
        pred_bboxes = self.bbox_pred(inst_features)

        # (L, B, C) -> (B, C, L) -> (B, C, H, W)
        inst_pixel_features = inst_pixel_features.permute(1, 2, 0).view(B, C, H, W)
        mask_pixel_features = mask_pixel_features.permute(1, 2, 0).view(B, C, H, W)

        results = {
            'logits': pred_logits,
            'objectness_scores': pred_scores,
            'kernels': {
                'instance_kernel': pred_inst_features,
                },
            'bboxes': {
                'instance_bboxes': pred_bboxes
                },
            'iams': {
                'instance_iams': inst_iam,
                },
            'inst_feats': {
                'instance_feats': inst_features
                },
            'inst_pixel_feats': inst_pixel_features,
            'mask_pixel_feats': mask_pixel_features
        }

        return results
    





# @HEADS.register(name="InstanceHead-v2.2-dual-update")
# class InstanceHead(nn.Module):
#     def __init__(self, 
#                  in_channels: int = 256, 
#                  num_convs: int = 4, 
#                  num_classes: int = 80, 
#                  kernel_dim: int = 256, 
#                  num_masks: int = 100, 
#                  num_groups: int = 1,
#                  activation: str = "softmax", 
#                  num_layers: int = 1,
#                  dim_feedforward: int = 2048,
#                  nhead: int = 8, 
#                  normalize_before: bool = False,
#                  dropout: float = 0.0):
#         super().__init__()
#         self.dim = in_channels
#         self.num_convs = num_convs
#         self.num_masks = num_masks
#         self.kernel_dim = kernel_dim
#         self.num_groups = num_groups
#         self.num_classes = num_classes + 1
#         self.activation = activation
#         self.scale_factor = 1

#         self.num_layers = 1
#         hidden_dim = self.dim * self.num_groups
        
#         # iam.
#         self.inst_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)

#         # positional encoding.
#         N_steps = hidden_dim // 2
#         self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

#         # ca.
#         self.transformer_instance_cross_attention_layers = nn.ModuleList([
#             CrossAttentionLayer(
#                 d_model=hidden_dim,
#                 nhead=nhead,
#                 dropout=dropout,
#                 normalize_before=normalize_before,
#             )
#             for _ in range(self.num_layers)
#         ])
        
#         # sa.
#         self.transformer_instance_self_attention_layers = nn.ModuleList([
#             SelfAttentionLayer(
#                 d_model=hidden_dim,
#                 nhead=nhead,
#                 dropout=dropout,
#                 normalize_before=normalize_before,
#             )
#             for _ in range(self.num_layers)
#         ])

#         # ffn.
#         self.transformer_instance_ffn_layers = nn.ModuleList([
#             FFNLayer(
#                 d_model=hidden_dim,
#                 dim_feedforward=dim_feedforward, 
#                 dropout=dropout, 
#                 normalize_before=normalize_before
#             )
#             for _ in range(self.num_layers)
#         ])


#         # ca - feats.
#         self.transformer_feats_cross_attention_layers = nn.ModuleList([
#             CrossAttentionLayer(
#                 d_model=hidden_dim,
#                 nhead=nhead,
#                 dropout=dropout,
#                 normalize_before=normalize_before,
#             )
#             for _ in range(self.num_layers)
#         ])

#         # ffn - feats.
#         self.transformer_feats_ffn_layers = nn.ModuleList([
#             FFNLayer(
#                 d_model=hidden_dim,
#                 dim_feedforward=dim_feedforward, 
#                 dropout=dropout, 
#                 normalize_before=normalize_before
#             )
#             for _ in range(self.num_layers)
#         ])

#         # ca - mask feats.
#         self.transformer_mask_cross_attention_layers = nn.ModuleList([
#             CrossAttentionLayer(
#                 d_model=hidden_dim,
#                 nhead=nhead,
#                 dropout=dropout,
#                 normalize_before=normalize_before,
#             )
#             for _ in range(num_layers)
#         ])
        
#         # ffn - mask feats.
#         self.transformer_mask_ffn_layers = nn.ModuleList([
#             FFNLayer(
#                 d_model=hidden_dim,
#                 dim_feedforward=dim_feedforward, 
#                 dropout=dropout, 
#                 normalize_before=normalize_before
#             )
#             for _ in range(num_layers)
#         ])

#         self.decoder_norm = nn.LayerNorm(hidden_dim)
#         self.query_embed = nn.Embedding(self.num_masks, hidden_dim)
#         self.prev_query_embed = nn.Embedding(self.num_masks, hidden_dim)

#         # Outputs
#         self.cls_score = nn.Linear(hidden_dim, self.num_classes)
#         self.inst_kernel = nn.Linear(hidden_dim, self.kernel_dim)
#         self.objectness = nn.Linear(hidden_dim, 1)
#         self.bbox_pred = nn.Linear(hidden_dim, 4)
        
#         self.prior_prob = 0.01
#         self._init_weights()
        
#         self.softmax_bias = nn.Parameter(torch.ones([1, ]))
#         print("v2.2-instance-dual-feat-update")


#     def _init_weights(self):
#         bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
#         init.normal_(self.cls_score.weight, std=0.01)
#         init.constant_(self.cls_score.bias, bias_value)
#         init.normal_(self.inst_kernel.weight, std=0.01)
#         init.constant_(self.inst_kernel.bias, 0.0)

    
#     def forward_one_layer(self, inst_features, pixel_features, mask_feats, query_embed, pos, i):
#         # ca.
#         pixel_features = self.transformer_feats_cross_attention_layers[i](
#             pixel_features, inst_features, 
#             memory_mask=None, 
#             memory_key_padding_mask=None, 
#             pos=query_embed, query_pos=pos
#             )
        
#         # ffn.
#         pixel_features = self.transformer_feats_ffn_layers[i](pixel_features)

#         # ca.
#         inst_features = self.transformer_instance_cross_attention_layers[i](
#             inst_features, pixel_features, 
#             memory_mask=None, 
#             memory_key_padding_mask=None, 
#             pos=pos, query_pos=query_embed
#             )
        
#         # sa.
#         inst_features = self.transformer_instance_self_attention_layers[i](
#             inst_features, tgt_mask=None, 
#             tgt_key_padding_mask=None, 
#             query_pos=query_embed
#             )
        
#         # ffn.
#         inst_features = self.transformer_instance_ffn_layers[i](inst_features)
        
        
#         mask_feats = self.transformer_mask_cross_attention_layers[i](
#             mask_feats, inst_features,
#             memory_mask=None,
#             memory_key_padding_mask=None,
#             pos=query_embed, query_pos=pos
#         )

#         mask_feats = self.transformer_mask_ffn_layers[i](mask_feats)

#         return inst_features, pixel_features, mask_feats


#     def forward(self, features, mask_feats, prev_inst_features=None):
#         inst_iam = self.inst_iam(features)

#         B, N, H, W = inst_iam.shape
#         C = features.size(1)
        
#         if self.activation == "softmax":
#             inst_iam_prob = F.softmax(inst_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
#         else:
#             raise NotImplementedError(f"No activation {self.activation} found!")
        
#         inst_features = torch.bmm(inst_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        
#         inst_features = inst_features.transpose(0, 1)
#         pixel_features = features.flatten(2).permute(2, 0, 1)
#         mask_feats = mask_feats.flatten(2).permute(2, 0, 1)

#         pos = self.pe_layer(features, None)
#         pos = pos.flatten(2).permute(2, 0, 1)
#         query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

#         if prev_inst_features is not None:
#             prev_inst_features = prev_inst_features.transpose(0, 1)
#             inst_features = torch.cat([inst_features, prev_inst_features], dim=0)
            
#             prev_query_embed = self.prev_query_embed.weight.unsqueeze(1).repeat(1, B, 1)
#             query_embed = torch.cat([query_embed, prev_query_embed], dim=0)

#         for i in range(self.num_layers):    
#             inst_features, pixel_features, mask_feats = self.forward_one_layer(
#                 inst_features, pixel_features, mask_feats, query_embed, pos, i
#             )

#         inst_features = inst_features[:self.num_masks]
#         inst_features = self.decoder_norm(inst_features)
#         inst_features = inst_features.transpose(0, 1)

#         # predictions.
#         pred_logits = self.cls_score(inst_features)
#         pred_inst_kernel = self.inst_kernel(inst_features)
#         pred_scores = self.objectness(inst_features)
#         pred_bboxes = self.bbox_pred(inst_features)

#         # (L, B, C) -> (B, C, L) -> (B, C, H, W)
#         pixel_features = pixel_features.permute(1, 2, 0).view(B, C, H, W)
#         mask_feats = mask_feats.permute(1, 2, 0).view(B, C, H, W)

#         results = {
#             'logits': pred_logits,
#             'objectness_scores': pred_scores,
#             'kernels': {
#                 'instance_kernel': pred_inst_kernel,
#                 },
#             'bboxes': {
#                 'instance_bboxes': pred_bboxes
#                 },
#             'iams': {
#                 'instance_iams': inst_iam,
#                 },
#             'inst_feats': {
#                 'instance_feats': inst_features
#                 },
#             'pixel_feats': pixel_features,
#             'mask_feats': mask_feats
#         }

#         return results
    

    


# @HEADS.register(name="InstanceHead-v2.2-two-way-attn")
# class InstanceHead(nn.Module):
#     def __init__(self, 
#                  in_channels: int = 256, 
#                  num_convs: int = 4, 
#                  num_classes: int = 80, 
#                  kernel_dim: int = 256, 
#                  num_masks: int = 100, 
#                  num_groups: int = 1,
#                  activation: str = "softmax", 
#                  num_layers: int = 1,
#                  dim_feedforward: int = 2048,
#                  nhead: int = 8, 
#                  normalize_before: bool = False,
#                  dropout: float = 0.0):
#         super().__init__()
#         self.dim = in_channels
#         self.num_convs = num_convs
#         self.num_masks = num_masks
#         self.kernel_dim = kernel_dim
#         self.num_groups = num_groups
#         self.num_classes = num_classes + 1
#         self.activation = activation
#         self.scale_factor = 1
#         self.nhead = nhead

#         self.num_layers = 1
#         hidden_dim = self.dim * self.num_groups
        
#         # iam.
#         self.inst_iam = IAM(self.dim, self.num_masks * self.num_groups, groups=self.num_groups)

#         # positional encoding.
#         N_steps = hidden_dim // 2
#         self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

#         # ca.
#         self.transformer_instance_cross_attention_layers = nn.ModuleList([
#             CrossAttentionLayer(
#                 d_model=hidden_dim,
#                 nhead=nhead,
#                 dropout=dropout,
#                 normalize_before=normalize_before,
#             )
#             for _ in range(self.num_layers)
#         ])
        
#         # sa.
#         self.transformer_instance_self_attention_layers = nn.ModuleList([
#             SelfAttentionLayer(
#                 d_model=hidden_dim,
#                 nhead=nhead,
#                 dropout=dropout,
#                 normalize_before=normalize_before,
#             )
#             for _ in range(self.num_layers)
#         ])

#         # ffn.
#         self.transformer_instance_ffn_layers = nn.ModuleList([
#             FFNLayer(
#                 d_model=hidden_dim,
#                 dim_feedforward=dim_feedforward, 
#                 dropout=dropout, 
#                 normalize_before=normalize_before
#             )
#             for _ in range(self.num_layers)
#         ])


#         # ca - feats.
#         self.transformer_feats_cross_attention_layers = nn.ModuleList([
#             CrossAttentionLayer(
#                 d_model=hidden_dim,
#                 nhead=nhead,
#                 dropout=dropout,
#                 normalize_before=normalize_before,
#             )
#             for _ in range(self.num_layers)
#         ])

#         # ffn - feats.
#         self.transformer_feats_ffn_layers = nn.ModuleList([
#             FFNLayer(
#                 d_model=hidden_dim,
#                 dim_feedforward=dim_feedforward, 
#                 dropout=dropout, 
#                 normalize_before=normalize_before
#             )
#             for _ in range(self.num_layers)
#         ])

#         self.decoder_norm = nn.LayerNorm(hidden_dim)
#         self.query_embed = nn.Embedding(self.num_masks, hidden_dim)
#         self.prev_query_embed = nn.Embedding(self.num_masks, hidden_dim)

#         # Outputs
#         self.cls_score = nn.Linear(hidden_dim, self.num_classes)
#         self.inst_kernel = nn.Linear(hidden_dim, self.kernel_dim)
#         self.objectness = nn.Linear(hidden_dim, 1)
#         self.bbox_pred = nn.Linear(hidden_dim, 4)
        
#         self.prior_prob = 0.01
#         self._init_weights()
        
#         self.softmax_bias = nn.Parameter(torch.ones([1, ]))
#         print("v2.2-instance-two-way-attn")


#     def _init_weights(self):
#         bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
#         init.normal_(self.cls_score.weight, std=0.01)
#         init.constant_(self.cls_score.bias, bias_value)
#         init.normal_(self.inst_kernel.weight, std=0.01)
#         init.constant_(self.inst_kernel.bias, 0.0)

    
#     def forward_one_layer(self, inst_features, pixel_features, query_embed, pos, i, attn_mask=None):
#         # ca.
#         pixel_features = self.transformer_feats_cross_attention_layers[i](
#             pixel_features, inst_features, 
#             memory_mask=None, 
#             memory_key_padding_mask=None, 
#             pos=query_embed, query_pos=pos
#             )
        
#         # ffn.
#         pixel_features = self.transformer_feats_ffn_layers[i](pixel_features)

#         # ca.
#         inst_features = self.transformer_instance_cross_attention_layers[i](
#             inst_features, pixel_features, 
#             memory_mask=None, 
#             memory_key_padding_mask=None, 
#             pos=pos, query_pos=query_embed
#             )
        
#         # sa.
#         inst_features = self.transformer_instance_self_attention_layers[i](
#             inst_features, tgt_mask=None, 
#             tgt_key_padding_mask=None, 
#             query_pos=query_embed
#             )
        
#         # ffn.
#         inst_features = self.transformer_instance_ffn_layers[i](inst_features)
        
#         return inst_features, pixel_features


#     def forward(self, features, mask_feats, prev_inst_features=None, attn_mask=None):
#         inst_iam = self.inst_iam(features)

#         B, N, H, W = inst_iam.shape
#         C = features.size(1)
        
#         if self.activation == "softmax":
#             inst_iam_prob = F.softmax(inst_iam.view(B, N, -1) + self.softmax_bias, dim=-1)
#         else:
#             raise NotImplementedError(f"No activation {self.activation} found!")
        
#         inst_features = torch.bmm(inst_iam_prob, features.view(B, C, -1).permute(0, 2, 1))
        
#         inst_features = inst_features.transpose(0, 1)
#         pixel_features = features.flatten(2).permute(2, 0, 1)

#         pos = self.pe_layer(features, None)
#         pos = pos.flatten(2).permute(2, 0, 1)
#         query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

#         if prev_inst_features is not None:
#             prev_inst_features = prev_inst_features.transpose(0, 1)
#             inst_features = torch.cat([inst_features, prev_inst_features], dim=0)
            
#             prev_query_embed = self.prev_query_embed.weight.unsqueeze(1).repeat(1, B, 1)
#             query_embed = torch.cat([query_embed, prev_query_embed], dim=0)

#         for i in range(self.num_layers):    
#             inst_features, pixel_features = self.forward_one_layer(
#                 inst_features, pixel_features, query_embed, pos, i, attn_mask
#             )

#         inst_features = inst_features[:self.num_masks]
#         inst_features = self.decoder_norm(inst_features)
#         inst_features = inst_features.transpose(0, 1)

#         # predictions.
#         pred_logits = self.cls_score(inst_features)
#         pred_inst_kernel = self.inst_kernel(inst_features)
#         pred_scores = self.objectness(inst_features)
#         pred_bboxes = self.bbox_pred(inst_features)

#         # (L, B, C) -> (B, C, L) -> (B, C, H, W)
#         pixel_features = pixel_features.permute(1, 2, 0).view(B, C, H, W)
#         # pixel_features = pixel_features.transpose(0, 1)
#         # pixel_features = self.mask_features_layer(pixel_features)
#         # pixel_features = pixel_features.permute(0, 2, 1).contiguous().view(B, C, H, W)

#         # attn_mask = self.get_attn_mask(pred_inst_kernel, pixel_features)

#         results = {
#             'logits': pred_logits,
#             'objectness_scores': pred_scores,
#             'kernels': {
#                 'instance_kernel': pred_inst_kernel,
#                 },
#             'bboxes': {
#                 'instance_bboxes': pred_bboxes
#                 },
#             'iams': {
#                 'instance_iams': inst_iam,
#                 },
#             'inst_feats': {
#                 'instance_feats': inst_features
#                 },
#             'pixel_feats': pixel_features,
#             # 'attn_mask': attn_mask
#         }

#         return results


#     def get_attn_mask(self, mask_embed, mask_features):
#         outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
#         outputs_mask = F.interpolate(outputs_mask, scale_factor=2, mode="bilinear", align_corners=False)

#         # outputs_mask.shape: b, q, h, w
#         attn_mask = F.pad(outputs_mask, (0, 0, 0, 0, 0, self.num_masks), "constant", 1)
#         attn_mask = (attn_mask < 0.).flatten(2)  # b, q, hw
#         invalid_query = attn_mask.all(-1, keepdim=True)  # b, q, 1
#         attn_mask = (~ invalid_query) & attn_mask  # b, q, hw
#         attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).flatten(0, 1)
#         attn_mask = attn_mask.detach()

#         return attn_mask
    
