import torch 
from torch import nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

import sys
sys.path.append("./")


class IAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super(IAM, self).__init__()
        self.iam_conv = nn.Conv2d(in_channels, out_channels, 
                                  kernel_size=kernel_size, stride=stride, padding=padding, 
                                  groups=groups)
        
        self.prior_prob = 0.01
        self.init_weights()

    def init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        init.normal_(self.iam_conv.weight, std=0.01)
        init.constant_(self.iam_conv.bias, bias_value)

    def forward(self, x):
        x = self.iam_conv(x)
        return x



class DeepIAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super(DeepIAM, self).__init__()
        self.iam_conv1 = nn.Conv2d(in_channels, in_channels, 
                                   kernel_size=kernel_size, stride=stride, padding=padding, 
                                   groups=groups)
        self.gelu = nn.GELU()
        self.iam_conv2 = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=kernel_size, stride=stride, padding=padding, 
                                   groups=groups)
        
        self.prior_prob = 0.01
        self.init_weights()

    def init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        init.normal_(self.iam_conv1.weight, std=0.01)
        init.constant_(self.iam_conv1.bias, bias_value)
        init.normal_(self.iam_conv2.weight, std=0.01)
        init.constant_(self.iam_conv2.bias, bias_value)

    def forward(self, x):
        x = self.iam_conv1(x)
        x = self.gelu(x)
        x = self.iam_conv2(x)
        return x


from einops import rearrange, einsum

class MHIAM(nn.Module):
    def __init__(self, embed_dim, n, num_heads, activation="softmax"):
        super(MHIAM, self).__init__()
        assert embed_dim % num_heads == 0, "in_channels must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.n = n
        self.activation = activation

        # self.in_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, groups=num_heads)

        self.attn_conv = nn.Conv2d(
            embed_dim, n * num_heads, kernel_size=3, padding=1, groups=num_heads
        )
        
        self.out_proj = nn.Linear(num_heads * embed_dim, embed_dim)

        self.bias = nn.Parameter(torch.ones([1]))

    # def forward(self, x):
    #     B, C, H, W = x.shape

    #     attn_scores = self.attn_conv(x).view(B, self.num_heads, self.n, H, W)
    #     attn_probs = F.softmax(attn_scores.view(B, self.num_heads, self.n, -1) + self.bias, dim=-1)  
    #     # (b, h, n, l)

    #     head_outs = torch.einsum('bhnl,bhdl->bhnd', attn_probs, 
    #                              x.view(B, self.num_heads, self.head_dim, H * W))

    #     # (b, hn, hw) (b, d, hw) (b, hn, d)
    #     # (b, h, n, hw) (b, h, d//h, hw) (b, h, n, d//h)

    #     cat_heads = rearrange(head_outs, 'b h n d -> b n (h d)')
    #     out = self.out_proj(cat_heads) # (b, n, d)
        
    #     attn_scores = rearrange(attn_scores, 'b h n hh ww -> b (h n) hh ww')

    #     return out, attn_scores



    def forward(self, x):
        B, C, H, W = x.shape
        
        attn_scores = self.attn_conv(x)
        attn_probs = F.softmax(attn_scores.view(B, self.num_heads * self.n, -1) + self.bias, dim=-1)  
        # (b, h, n, l)

        head_outs = torch.einsum('bnl,bdl->bnd', attn_probs, x.view(B, C, -1))
        head_outs = head_outs.view(B, self.num_heads, self.n, -1)

        # (b, hn, hw) (b, d, hw) (b, hn, d)

        cat_heads = rearrange(head_outs, 'b h n d -> b n (h d)')
        out = self.out_proj(cat_heads) # (b, n, d)

        return out, attn_scores
