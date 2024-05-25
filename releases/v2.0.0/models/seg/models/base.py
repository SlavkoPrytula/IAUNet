import torch
from torch import nn

# from models.seg.heads.instance_head import InstanceBranch, PriorInstanceBranch
# from models.seg.heads.mask_head import MaskBranch
# from models.seg.modules.mixup import MixUpScaler

from configs import cfg

from abc import ABC, abstractmethod

class PyramidPooling(nn.Module):
    def __init__(self, kernel_strides_map, n_filters=64):
        super(PyramidPooling, self).__init__()
        self.kernel_strides_map = kernel_strides_map
        self.n_filters = n_filters
        self.interp_block1 = Interpolation(1, self.kernel_strides_map, self.n_filters)
        self.interp_block2 = Interpolation(2, self.kernel_strides_map, self.n_filters)
        self.interp_block3 = Interpolation(3, self.kernel_strides_map, self.n_filters)
        self.interp_block4 = Interpolation(4, self.kernel_strides_map, self.n_filters)
        self.interp_block5 = Interpolation(5, self.kernel_strides_map, self.n_filters)
        
    def forward(self, x):
        interp_out1 = self.interp_block1(x)
        interp_out2 = self.interp_block2(x)
        interp_out3 = self.interp_block3(x)
        interp_out4 = self.interp_block4(x)
        interp_out5 = self.interp_block5(x)
        x = torch.cat(
            [x, interp_out5, interp_out4, interp_out3, interp_out2, interp_out1], dim=1
        )
        return x
    
    
class Interpolation(nn.Module):
    def __init__(self, level, kernel_strides_map, n_filters=64):
        super(Interpolation, self).__init__()
        self.level = level
        self.kernel = kernel_strides_map[level]
        self.stride = kernel_strides_map[level]
        self.n_filters = n_filters
        self.interp = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.kernel, stride=self.stride),
            nn.Conv2d(
                self.n_filters, self.n_filters // 4, kernel_size=1, stride=1, bias=False
            ),
#             nn.BatchNorm2d(
#                 num_features=self.n_filters // 4, momentum=0.95, eps=1e-5, affine=False
#             ),
            nn.BatchNorm2d(self.n_filters // 4),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=int(16 / (2 ** (self.level - 1)))),
        )
    def forward(self, x):
        x = self.interp(x)
        return x
    
    
class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x   
    
    
class SE_block(nn.Module):
    """squeeze and excitation block"""
    def __init__(self, num_features, reduction_factor=2):
        super(SE_block, self).__init__()
        # squeeze block
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # excitation block
        self.excite = nn.Sequential(
            nn.Linear(num_features, num_features // reduction_factor),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // reduction_factor, num_features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch, channel, _, _ = x.size()
        squeeze_res = self.squeeze(x).view(batch, channel)
        excite_res = self.excite(squeeze_res)
        f_scale = excite_res.view(batch, channel, 1, 1)
        return x * f_scale



class SparseSEUnet(nn.Module, ABC):
    def __init__(
        self,
        cfg: cfg,
        n_filters=64,
        pyramid_pooling=True,
        n_pp_features=144,
    ):
        super(SparseSEUnet, self).__init__()
        self.cfg = cfg
        self.n_input_channels = cfg.model.in_channels
        self.n_output_channels = cfg.model.out_channels
        self.n_levels = cfg.model.n_levels

        self.n_filters = n_filters
        self.n_pp_features = n_pp_features
        self.pyramid_pooling = pyramid_pooling
        self.kernel_strides_map = {1: 16, 2: 8, 3: 4, 4: 2, 5: 1}

        self.down_conv_layers = nn.ModuleList([])
        self.down_pp_layers = nn.ModuleList([])
        self.down_se_blocks = nn.ModuleList([])
        self.up_se_blocks = nn.ModuleList([])
        self.pp_se_blocks = nn.ModuleList([])
        
        self.middleConv = DoubleConv(
            self.n_filters * 2, self.n_filters, kernel_size=3, stride=1
            )
        self.middleSE = SE_block(num_features = self.n_filters)

        self.up_conv_layers = nn.ModuleList([])
        for _ in range(self.n_levels):
            # down convolution
            if len(self.down_conv_layers) == 0:
                downconv = DoubleConv(self.n_input_channels, self.n_filters)
            elif len(self.down_conv_layers) == 1:
                downconv = DoubleConv(self.n_filters, self.n_filters)
            else:
                downconv = DoubleConv(self.n_filters * 2, self.n_filters)
            self.down_conv_layers.append(downconv)
            # SE blocks following the downconv 
            down_se = SE_block(num_features = self.n_filters)
            self.down_se_blocks.append(down_se)
            # up convolution
            if len(self.up_conv_layers) == 0:
                # if self.coord_conv:
                upconv = DoubleConv(self.n_filters+2, self.n_filters)
                # else:
                # upconv = DoubleConv(self.n_filters, self.n_filters)
            else:
                # if self.coord_conv:
                upconv = DoubleConv(
                    (self.n_filters // 4) * 5 + 2 * self.n_filters+2, self.n_filters
                )
                # else:
                # upconv = DoubleConv(
                #     (self.n_filters // 4) * 5 + 2 * self.n_filters, self.n_filters
                # )

            self.up_conv_layers.append(upconv)

             # SE blocks following the upconv 
            up_se = SE_block(num_features = self.n_filters)            
            self.up_se_blocks.append(up_se)
            
            # down pyramid
            if self.pyramid_pooling:
                pplayer = PyramidPooling(
                    kernel_strides_map=self.kernel_strides_map, n_filters=self.n_filters
                )
                self.down_pp_layers.append(pplayer)
                 # SE blocks following the pp block
                pp_se = SE_block(num_features = self.n_pp_features)            
                self.pp_se_blocks.append(pp_se)


        for modules in [self.down_conv_layers, self.down_se_blocks,
                        self.up_conv_layers, self.up_se_blocks, 
                        self.down_pp_layers, self.pp_se_blocks]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)
                
        
    # @torch.no_grad()
    def compute_coordinates_linspace(self, x):
        # linspace is not supported in ONNX
        h, w = x.size(2), x.size(3)
        y_loc = torch.linspace(-1, 1, h, device=x.device)
        x_loc = torch.linspace(-1, 1, w, device=x.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)


    # @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        # y_loc = -1.0 + 2.0 * torch.arange(h, device=x.device) / (h - 1)
        # x_loc = -1.0 + 2.0 * torch.arange(w, device=x.device) / (w - 1)
        # y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        # y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        # x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        # locations = torch.cat([x_loc, y_loc], 1)
        # return locations.to(x)
    
        x_loc = torch.linspace(-1, 1, h, device=x.device)
        y_loc = torch.linspace(-1, 1, w, device=x.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x_loc, y_loc], 1)

        return coord_feat
        

    @abstractmethod
    def forward(self, x, idx=None):
        ...
