import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath



class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = torch.cat([x, out], 1)
        return out

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.dense_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.dense_layers(x)



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))



class ConvNeXtBlock(nn.Module):
    def __init__(self, in_dim, out_dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(in_dim, in_dim, kernel_size=7, padding=3, groups=in_dim)  # depthwise conv
        self.norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.pwconv1 = nn.Linear(in_dim, 4 * in_dim)  # first pointwise conv, expanding channels
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * in_dim, out_dim)  # second pointwise conv, contracting to out_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if in_dim != out_dim:
            self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)

        input = self.proj(input)
        x = input + self.drop_path(x)
        return x



class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim_feedforward=2048, nhead=8, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Flatten the feature map from (H, W) to (H*W) and treat it as the sequence length
        self.self_attn = nn.MultiheadAttention(in_channels, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(in_channels, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, in_channels)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        # Additional layer to control output channels
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def _forward_impl(self, x):
        # Change input shape to [batch_size * H * W, in_channels] for self attention
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * h * w, c)

        # Self-attention
        x2 = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x2, x2, x2)[0])

        # Feedforward
        x2 = self.norm2(x)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x = x + self.dropout2(x2)

        # Reshape x back to [batch, channels, height, width]
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)

        # Apply the final convolution to adjust the output channels
        x = self.final_conv(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)




if __name__ == "__main__":
    in_channels = 64
    out_channels = 128

    block = TransformerBlock(in_channels, out_channels)
    x = torch.rand(1, in_channels, 128, 128)
    out = block(x)
    print(out.shape)

    from thop import profile
    ops, params = profile(block, inputs=(x,), verbose=True)
    print(params)