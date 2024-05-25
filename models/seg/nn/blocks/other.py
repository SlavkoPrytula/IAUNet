import torch 
from torch import nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


# class Block(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         # self.dwconv = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
#         # # self.norm = nn.LayerNorm(dim, eps=1e-6)
#         # self.norm = nn.BatchNorm2d(dim),
#         # # self.pwconv1 = nn.Linear(dim, 4 * dim)
#         # self.pwconv1 = nn.Conv2d(dim, dim * 4, kernel_size=(1, 1))
#         # self.act = nn.GELU()
#         # # self.pwconv2 = nn.Linear(4 * dim, dim)
#         # self.pwconv2 = nn.Conv2d(dim * 4, dim, kernel_size=(1, 1))

#         self.block = nn.Sequential(
#             nn.Conv2d(dim, dim, 7, 1, 3, groups=dim),
#             nn.GELU(),
#             nn.BatchNorm2d(dim),
#             nn.Conv2d(dim, dim * 4, kernel_size=(1, 1)),
#             nn.GELU(),
#             nn.BatchNorm2d(dim * 4),
#             nn.Conv2d(dim * 4, dim, kernel_size=(1, 1)),
#             nn.GELU(),
#             nn.BatchNorm2d(dim),
#             nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         input = x
#         x = self.block(x)
#         # x = input + x

#         return x
    

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)

        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N ,H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = x.permute(0, 3, 1, 2)
        x = input + x

        return x
    


class DWCFusion(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.dwconv = nn.Conv2d(ch_in, ch_in, 7, 1, 3, groups=ch_in)
        self.norm = nn.LayerNorm(ch_in, eps=1e-6)
        self.pwconv1 = nn.Linear(ch_in, 4 * ch_in)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * ch_in, ch_out)

    def forward(self, x):
        # input = x
        x = self.dwconv(x)

        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N ,H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = x.permute(0, 3, 1, 2)
        # x = input + x

        return x
    


class FusionConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(FusionConv, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(ch_in, ch_in, kernel_size=7, stride=1, padding=3, groups=ch_in, bias=True),
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        x = self.conv(x)
        return x




# class Block(nn.Module):
#     def __init__(self, dim, dpr=0., init_value=1e-6):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
#         self.norm = nn.LayerNorm(dim, eps=1e-6)
#         self.pwconv1 = nn.Linear(dim, 4 * dim)
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(4 * dim, dim)
#         self.gamma = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True) if init_value > 0 else None
#         self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()

#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N ,H, W, C]
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)

#         if self.gamma is not None:
#             x = self.gamma * x

#         x = x.permute(0, 3, 1, 2)
#         x = input + self.drop_path(x)
#         return x



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    

class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super(CMUNeXtBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # deep wise
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ) for i in range(depth)]
        )
        self.up = conv_block(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x
    


class DWSConv2d(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 1):
        super(DWSConv2d, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class FFN(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x2 = self.linear2(self.dropout(self.relu(self.linear1(x))))
        x = x + self.dropout(x2)
        x = self.norm(x)
        return x


class IAMClsHead(nn.Module):
    def __init__(self, input_dim, output_dim, num_convs):
        super(IAMClsHead, self).__init__()

        self.bn = nn.BatchNorm2d(input_dim)
        self.relu = nn.ReLU(inplace=True)
        # self.iam_head = _make_stack_3x3_convs(num_convs, input_dim, output_dim)

        self.cls_head = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, 3, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, output_dim, 3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d(32)
        # self.fc = nn.Linear(1024, 1024)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.cls_head(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), x.size(1), -1)
        # x = F.relu_(self.fc(x))

        return x
        