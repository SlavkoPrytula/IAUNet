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
    def __init__(self, in_channels, out_channels, groups=1):
        super(DeepIAM, self).__init__()
        self.num_convs = 2
        
        convs = []
        for _ in range(self.num_convs):
            convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups))
            convs.append(nn.BatchNorm2d(out_channels))
            convs.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        self.convs = nn.Sequential(*convs)
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)

        self._init_weights()

    def _init_weights(self):
        for m in self.convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        c2_msra_fill(self.projection)

    def forward(self, x):
        x = self.convs(x)
        x = self.projection(x)
        
        return x



import math
class AttentionIAM(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        in_channels, out_channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = in_channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                in_channels % num_head_channels == 0
            ), f"q,k,v channels {in_channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = in_channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = nn.GroupNorm(32, in_channels)
        self.qkv = nn.Conv1d(in_channels, in_channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out((x + h).reshape(b, c, *spatial))
        return h


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)



if __name__ == '__main__':
    x = torch.randn(2, 32, 64, 64)
    iam = AttentionIAM(32, 10)
    out = iam(x)
    print(out.shape)