import torch
from torch import nn
from torch.nn import functional as F

import sys
sys.path.append('.')

from models.seg.layers import get_norm
# from models.seg.heads.common import c2_msra_fill

import fvcore.nn.weight_init as weight_init


class Conv2d(nn.Conv2d):
    """
    A wrapper around :class:`torch.Conv2d` to support more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0 and self.training:
            # https://github.com/pytorch/pytorch/issues/12013
            assert not isinstance(
                self.norm, torch.nn.SyncBatchNorm
            ), "SyncBatchNorm does not support empty inputs!"

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class GCN(nn.Module):
    def __init__(self, in_channels):
        super(GCN, self).__init__()

        # query, key, value transformations
        self.query_transform = Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.key_transform = Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.value_transform = Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.output_transform = Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

        # scale, blocker
        self.scale = 1.0 / (in_channels ** 0.5)
        self.blocker = nn.BatchNorm2d(in_channels, eps=1e-04) # should be zero initialized

        self.init_weights()

    def init_weights(self):
        for layer in [self.query_transform, self.key_transform, self.value_transform, self.output_transform]:
            weight_init.c2_msra_fill(layer)

    def forward(self, x):
        B,C,H,W = x.shape
        # x: B,C,H,W

        # query: (256, H, W) -> qt() -> (256, H, W) -> (256, [HW]) -> ([HW], 256)
        # key:   (256, H, W) -> kt() -> (256, H, W) -> (256, [HW])
        # value: (256, H, W) -> vt() -> (256, H, W) -> (256, [HW]) -> ([HW], 256)

        # w:     ([HW], 256) x (256, [HW]) -> ([HW], [HW]) = W
        # w:     W = softmax(W, dim=-1)    -> ([HW], [HW])
        
        # rel:   ([HW], [HW]) x ([HW], 256) -> ([HW], 256) -> (256, [HW]) -> (256, H, W)
        # out:   (256, H, W) -> ot() -> (256, H, W) -> BN()

        # x_query: B,C,HW
        x_query = self.query_transform(x).view(B, C, -1)
        # x_query: B,HW,C
        x_query = torch.transpose(x_query, 1, 2)
        
        # x_key: B,C,HW
        x_key = self.key_transform(x).view(B, C, -1)
        
        # x_value: B,C,HW
        x_value = self.value_transform(x).view(B, C, -1)
        # x_value: B,HW,C
        x_value = torch.transpose(x_value, 1, 2)
        
        # W = Q^T K: B,HW,HW
        x_w = torch.matmul(x_query, x_key) * self.scale
        x_w = F.softmax(x_w, dim=-1)
        
        # x_relation = WV: B,HW,C
        x_relation = torch.matmul(x_w, x_value)
        # x_relation = B,C,HW
        x_relation = torch.transpose(x_relation, 1, 2)
        # x_relation = B,C,H,W
        x_relation = x_relation.view(B,C,H,W)

        x_relation = self.output_transform(x_relation)
        x_relation = self.blocker(x_relation)

        x = x + x_relation

        return x


class Model(nn.Module):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, input_shape):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super(Model, self).__init__()

        # fmt: off
        num_classes       = 1
        conv_dims         = 64
        self.norm         = "GN"
        num_conv          = 4
        input_channels    = input_shape[1]
        cls_agnostic_mask = True
        # fmt: on

        self.conv_norm_relu_instance = nn.ModuleList([])
        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.conv_norm_relu_instance.append(conv)

        self.conv_norm_relu_occluder = nn.ModuleList([])
        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.conv_norm_relu_occluder.append(conv)


        self.gcn_1 = GCN(input_channels)
        self.gcn_2 = GCN(input_channels)

        for layer in self.conv_norm_relu_instance + self.conv_norm_relu_occluder:
            weight_init.c2_msra_fill(layer)


    def forward(self, x):
        B, C, H, W = x.size()
        x_orig = x.clone()  # save original feats

        # (conv) --> (conv + gcn) --> (conv) --> (conv)
        #     0                1          2          3

        for cnt, layer in enumerate(self.conv_norm_relu_occluder):
            x = layer(x)  # conv layer

            if cnt == 1 and len(x) != 0:
                x = self.gcn_1(x)

        x_occluder = x.clone()
        
        x = x_orig + x  # add occluder feats

        for cnt, layer in enumerate(self.conv_norm_relu_instance):
            x = layer(x)
            if cnt == 1 and len(x) != 0:
                x = self.gcn_2(x)

        x_instance = x.clone()

        return x_instance, x_occluder


if __name__ == "__main__":
    x = torch.rand(2, 64, 512, 512)
    print(x.shape)

    model = Model(x.shape)

    out = model(x)
    print(out[0].shape)