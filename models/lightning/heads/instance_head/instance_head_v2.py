import einops
import torch 
from torch import nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

import sys
sys.path.append('.')

from .iam import (IAM)
from ..common import _make_stack_3x3_convs
from models.seg.nn.blocks import (DoubleConv_v1, DoubleConv_v3_1)
from models.seg.nn.blocks import MLP

from configs import cfg
from utils.registry import HEADS


class IABlock(nn.Module):
    def __init__(self, 
                 in_channels: int = 256, 
                 query_dim: int = 256, 
                 num_queries: int = 100, 
                 activation: str = "softmax",
                 ):
        super().__init__()
        self.dim = in_channels
        self.num_queries = num_queries
        self.query_dim = query_dim
        self.activation = activation
        
        self.iam_conv = IAM(self.dim, self.num_queries)
        
        expand_dim = self.dim
        self.fc = nn.Linear(expand_dim, expand_dim)
        self.proj = nn.Linear(expand_dim, self.query_dim)

        self._init_weights()
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))


    def _init_weights(self):
        init.normal_(self.proj.weight, std=0.01)
        init.constant_(self.proj.bias, 0.0)
        c2_xavier_fill(self.fc)


    def forward(self, features):
        iam = self.iam_conv(features)
        B, N, H, W = iam.shape
        C = features.size(1)

        if self.activation == "softmax":
            iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)
        else:
            raise NotImplementedError(f"No activation {self.activation} found!")
        
        inst_features = torch.bmm(iam_prob, features.view(B, C, -1).permute(0, 2, 1))

        inst_features = F.relu_(self.fc(inst_features))
        inst_features = self.proj(inst_features)

        return inst_features, iam