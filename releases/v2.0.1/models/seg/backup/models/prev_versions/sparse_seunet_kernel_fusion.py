import torch
from torch import nn

from models.seg.heads.instance_head import InstanceBranch
from models.seg.heads.mask_head import MaskBranch
from models.seg.modules.mixup import MixUpScaler
from models.seg.decoders import BaseDecoder


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
        #print(f"pp in size: {x.shape}")
        interp_out1 = self.interp_block1(x)
        interp_out2 = self.interp_block2(x)
        interp_out3 = self.interp_block3(x)
        interp_out4 = self.interp_block4(x)
        interp_out5 = self.interp_block5(x)
        x = torch.cat(
            [x, interp_out5, interp_out4, interp_out3, interp_out2, interp_out1], dim=1
        )
        #print(f"pp out size: {x.shape}")
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
        #print(f'squeeze_res: {squeeze_res.shape}')
        excite_res = self.excite(squeeze_res)
        #print(f'excite_res: {excite_res.shape}')
        f_scale = excite_res.view(batch, channel, 1, 1)
        #print(f'f_scale: {f_scale.shape}')
        return x * f_scale



class SparseSEUnet(nn.Module):
    def __init__(
        self,
        n_input_channels,
        n_output_channels,
        n_levels,
        n_filters=64,
        pyramid_pooling=True,
        n_pp_features=144,
    ):
        super(SparseSEUnet, self).__init__()
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        self.n_levels = n_levels
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
#         self.finalconv = nn.Conv2d(
#             (self.n_filters // 4) * 5 + 2 * self.n_filters,
#             self.n_output_channels,
#             kernel_size=1,
#             stride=1,
#         )
        
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
                upconv = DoubleConv(self.n_filters+2, self.n_filters)
            else:
                upconv = DoubleConv(
                    (self.n_filters // 4) * 5 + 2 * self.n_filters+2, self.n_filters
                )
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
                
        self.scale = MixUpScaler(scale_factor=2)
#         self.decoders = nn.ModuleList([
#             BaseDecoder(in_channels=208, num_masks=25),
#             BaseDecoder(in_channels=208, num_masks=25),
# #             BaseDecoder(in_channels=208, num_masks=25),
# #             BaseDecoder(in_channels=208, num_masks=25),
# #             BaseDecoder(in_channels=208, num_masks=25),
#         ])

        self.decoder = BaseDecoder(in_channels=128, num_masks=25)
        
#         self.ppm = UPerDecoder(
#                in_dim=[64, 64, 64, 64, 64],
#                ppm_pool_scale=[1, 2, 3, 4, 6],
#                ppm_dim=512,
#                fpn_out_dim=512
#         )
        
        
#         self.cyto_conv = nn.Conv2d(
#             (self.n_filters // 4) * 5 + 2 * self.n_filters,
#             1,
#             kernel_size=1,
#             stride=1,
#         )
#         self.dx_grad_conv = nn.Conv2d(
#             (self.n_filters // 4) * 5 + 2 * self.n_filters,
#             1,
#             kernel_size=1,
#             stride=1,
#         )
#         self.dy_grad_conv = nn.Conv2d(
#             (self.n_filters // 4) * 5 + 2 * self.n_filters,
#             1,
#             kernel_size=1,
#             stride=1,
#         )
        
        self.flow_conv = nn.Conv2d(
            (self.n_filters // 4) * 5 + 2 * self.n_filters,
            2,
            kernel_size=1,
            stride=1,
        )
        
        self.mask_branch = nn.ModuleList([
            MaskBranch(208),
            MaskBranch(208+128),
            MaskBranch(208+128),
            MaskBranch(208+128),
        ])
        
        self.instance_branch = nn.ModuleList([
            InstanceBranch(in_channels=208, kernel_dim=128, num_masks=50),
            InstanceBranch(in_channels=208, kernel_dim=128, num_masks=50),
            InstanceBranch(in_channels=208, kernel_dim=128, num_masks=50),
            InstanceBranch(in_channels=208, kernel_dim=128, num_masks=50),
        ])
        
        self.instance_conv = nn.ModuleList([
            nn.Conv1d(256, 128, kernel_size=1, stride=1),
            nn.Conv1d(256, 128, kernel_size=1, stride=1),
            nn.Conv1d(256, 128, kernel_size=1, stride=1),
        ])
            
                
    def forward(self, x):
        down_conv_out_tensors = []
        down_pp_out_tensors = []
        down_pool_out_tensors = []
        
        # go down
        for i in range(self.n_levels):
            x = self.down_conv_layers[i](x)
            x = self.down_se_blocks[i](x)
            down_conv_out_tensors.append(x)
            if self.pyramid_pooling:
                x_pp = self.down_pp_layers[i](x)
                x_pp = self.pp_se_blocks[i](x_pp)
                down_pp_out_tensors.append(x_pp)
            x = nn.MaxPool2d(2)(x)
            
            down_pool_out_tensors.append(x)
            # Skip connection if required
            if i > 0:
                x = nn.MaxPool2d(2)(down_pool_out_tensors[-2])
                x = torch.cat([x, down_pool_out_tensors[-1]], dim=1)
                
                
        # middle
        x = self.middleConv(x)
        x = self.middleSE(x)
        
        
        # go up
        def go_up(x):
            iams = []
            
            for i in range(self.n_levels):
                coord_features = self.compute_coordinates(x)
                x = torch.cat([coord_features, x], dim=1)
                    
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
                x = self.up_conv_layers[i](x)
                x = self.up_se_blocks[i](x)
                
                if self.pyramid_pooling:
                    x = torch.cat([x, down_pp_out_tensors[-(i + 1)]], dim=1)
                else:
                    x = torch.cat([x, down_conv_out_tensors[-(i + 1)]], dim=1)
                
                if i != 0:
                    mb = nn.UpsamplingBilinear2d(scale_factor=2)(mb)    # (1, 128, 128, 128)
                    
                    mb = torch.cat([x, mb], dim=1)
                    mb = self.mask_branch[i](mb)     
                else:
                    mb = self.mask_branch[i](x) 


                    
                if i != 0:
                    logits, kernel_upper, scores, iam = self.instance_branch[i](x)   
                    
                    kernel = torch.cat([kernel, kernel_upper], dim=-1)
                    kernel = self.instance_conv[i-1](kernel.transpose(1, 2)).transpose(1, 2)
                else:
                    _, kernel, _, _ = self.instance_branch[i](x)

            return x, mb, (logits, kernel, scores, iam)
    
        
    
        # cyto
        x, mb, (logits, kernel, scores, iam) = go_up(x)
        x = self.flow_conv(x)
        
#         print(x.shape)
#         print(mb.shape)
#         print(kernel.shape)
        
        
        # Predicting instance masks
        N = kernel.shape[1]  # num_masks
        
        B, C, H, W = mb.shape
        mask_features = mb.view(B, C, -1)   # (B, C, H, W) -> (B, C, [HW])
        masks = torch.matmul(
            kernel,    # (B, N, 128)
            mask_features   # (B, 128, [HW])
        ) # -> (B, N, [HW])
        masks = masks.view(B, N, H, W)  # (B, N, [HW]) -> (B, N, H, W)
        
        
        
        flow_pred = nn.Tanh()(x)
        
    
        return flow_pred, logits, masks, scores, masks
    
    
    @torch.no_grad()
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

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = -1.0 + 2.0 * torch.arange(h, device=x.device) / (h - 1)
        x_loc = -1.0 + 2.0 * torch.arange(w, device=x.device) / (w - 1)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)
    