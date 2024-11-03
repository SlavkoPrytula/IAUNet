import torch
import torchvision.models as models
from torch import nn

import sys
sys.path.append("./")

from utils.registry import MODELS
from models.seg.nn.blocks import DoubleConv, SE_block, DoubleConvModule
from models.seg.nn.blocks import PyramidPooling_v5


@MODELS.register(name="UNet")
class UNet(nn.Module):
    def __init__(self, 
        num_stages=5,
        out_indices=(0, 1, 2, 3, 4),
        pyramid_pooling=False,
        pp_embed_dim=128,
        embed_dims=[128, 128, 128, 128, 128],
        depths=[1, 1, 1, 1, 1]
        ):
        super().__init__()
        self.in_channels = 3
        self.num_stages = num_stages
        self.out_indices = out_indices
        self.pyramid_pooling = pyramid_pooling
        self.embed_dims = embed_dims
        self.pp_embed_dim = pp_embed_dim
        assert self.num_stages <= len(self.out_indices)

        self.down_conv_layers = nn.ModuleList([])
        self.down_pp_layers = nn.ModuleList([])
        self.down_se_blocks = nn.ModuleList([])
        self.pp_se_blocks = nn.ModuleList([])


        for i in range(self.num_stages):
            # down convolution
            if i == 0:
                in_channels = self.in_channels
                downconv = nn.Sequential(
                    nn.Conv2d(in_channels, self.embed_dims[i], kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.embed_dims[i]),
                    nn.ReLU(inplace=True)
                )
            elif i == 1:
                in_channels = embed_dims[i-1]
                downconv = nn.Sequential(
                    nn.Conv2d(in_channels, self.embed_dims[i], kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.embed_dims[i]),
                    nn.ReLU(inplace=True)
                )
            else:
                in_channels = embed_dims[i-1] + embed_dims[i-2]
                downconv = DoubleConvModule(in_channels, self.embed_dims[i], depth=depths[i])

            # downconv = DoubleConvModule(in_channels, self.embed_dims[i], depth=depths[i])
            # downconv = DoubleConv(in_channels, self.embed_dims[i])
            self.down_conv_layers.append(downconv)
           
            down_se = SE_block(num_features=self.embed_dims[i])
            self.down_se_blocks.append(down_se)


            # down pyramid
            if self.pyramid_pooling:
                pplayer = PyramidPooling_v5(in_channels=self.embed_dims[i], 
                                            pool_sizes=[1, 2, 4, 8], 
                                            out_channels=self.pp_embed_dim, 
                                            expand=1)
                self.down_pp_layers.append(pplayer)

                pp_se = SE_block(num_features=self.pp_embed_dim)            
                self.pp_se_blocks.append(pp_se)

        self._init_weights()


    def _init_weights(self):
        for modules in [self.down_conv_layers, self.down_se_blocks,
                        self.down_pp_layers, self.pp_se_blocks]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)


    def forward(self, x):
        outputs = []

        down_pool_out_tensors = []
        
        # go down
        for i in range(self.num_stages):
            x = self.down_conv_layers[i](x)
            x = self.down_se_blocks[i](x)
            
            if self.pyramid_pooling:
                x_pp = self.down_pp_layers[i](x)
                x_pp = self.pp_se_blocks[i](x_pp)

            outputs.append(x)
            
            x = nn.MaxPool2d(2)(x)
            
            down_pool_out_tensors.append(x)
            # residual connection if required
            if i > 0:
                x = nn.MaxPool2d(2)(down_pool_out_tensors[-2])
                x = torch.cat([x, down_pool_out_tensors[-1]], dim=1)

        return outputs

