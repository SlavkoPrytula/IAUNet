import torch 
from torch import nn
from torch.nn import init
import numpy as np

from timm.models.layers import trunc_normal_, DropPath


def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def kaiming_init(module: nn.Module,
                 a: float = 0,
                 mode: str = 'fan_out',
                 nonlinearity: str = 'relu',
                 bias: float = 0,
                 distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)

        
def c2_msra_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # pyre-fixme[6]: For 1st param expected `Tensor` but got `Union[Module, Tensor]`.
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
        #  torch.Tensor]`.
        nn.init.constant_(module.bias, 0)



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
    


class FusionConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(FusionConv, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(ch_in, ch_in, kernel_size=7, stride=1, padding=3, groups=ch_in, bias=True),
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
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
    


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
        

def _make_stack_3x3_convs(num_convs, in_channels, out_channels):
    convs = []
    for _ in range(num_convs):
        convs.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        convs.append(nn.BatchNorm2d(out_channels))
        convs.append(nn.ReLU(inplace=True))
        in_channels = out_channels
    return nn.Sequential(*convs)



if __name__ == "__main__":
    block = DWCFusion(32, 16)
    x = torch.rand(1, 32, 10, 10)
    out = block(x)
    print(out.shape)
