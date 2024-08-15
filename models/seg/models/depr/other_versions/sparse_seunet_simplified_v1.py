import torch
from torch import nn
import importlib.util

# Specify the path to instance_head.py
# instance_head_file = "/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/models/seg/heads/instance_head/instance_head.py"

# # Load the instance_head.py file as a module
# spec = importlib.util.spec_from_file_location('instance_head', instance_head_file)
# instance_head = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(instance_head)

# InstanceBranch = instance_head.InstanceBranch
# PriorInstanceBranch = instance_head.PriorInstanceBranch
# GroupInstanceBranch = instance_head.GroupInstanceBranch

import sys
sys.path.append("./")

from models.seg.heads.instance_head import InstanceBranch, PriorInstanceBranch
from models.seg.heads.mask_head import MaskBranch

# from ..heads.instance_head import InstanceBranch, PriorInstanceBranch, GroupInstanceBranch
# from ..heads.mask_head import MaskBranch

# from models.seg.modules.mixup import MixUpScaler
# from models.seg.models.base import SparseSEUnet as BaseModel
from models.seg.models.base import DoubleConv, SE_block

from configs import cfg


# class SparseSEUnet(nn.Module):
#     def __init__(
#         self,
#         cfg: cfg,
#         n_filters=64,
#         pyramid_pooling=True,
#         n_pp_features=144,
#     ):
#         super(SparseSEUnet, self).__init__()  

#         self.cfg = cfg
#         self.n_input_channels = cfg.model.in_channels
#         self.n_output_channels = cfg.model.out_channels
#         self.n_levels = cfg.model.n_levels

#         self.n_filters = n_filters
#         self.n_pp_features = n_pp_features
#         self.pyramid_pooling = pyramid_pooling
#         self.kernel_strides_map = {1: 16, 2: 8, 3: 4, 4: 2, 5: 1}
        
#         self.coord_conv = cfg.model.coord_conv
#         self.multi_level = cfg.model.multi_level
#         self.kernel_dim = cfg.model.kernel_dim
#         self.num_masks = cfg.model.num_masks
#         self.num_convs = cfg.model.num_convs
        
#         # mask branch.
#         self.mask_branch = nn.ModuleList([])
#         self.mask_branch.append(MaskBranch(320))
#         # for i in range(self.n_levels):
#         #     if i == 0:
#         #         self.mask_branch.append(MaskBranch(208, num_convs=self.num_convs))
#         #     else:
#         #         self.mask_branch.append(MaskBranch(208+128, num_convs=self.num_convs))
        
#         # instance features.
#         self.prior_instance_branch = nn.ModuleList([])
#         self.prior_instance_branch.append(PriorInstanceBranch(in_channels=320, out_channels=256, num_convs=4))
#         # for i in range(self.n_levels):
#         #     if i == 0:
#         #         self.prior_instance_branch.append(PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=self.num_convs))
#         #     else:
#         #         self.prior_instance_branch.append(PriorInstanceBranch(in_channels=464, out_channels=256, num_convs=self.num_convs))

#         # instance branch.
#         self.instance_branch = InstanceBranch(dim=256, kernel_dim=self.kernel_dim, num_masks=self.num_masks)



#         self.down_conv_layers = nn.ModuleList([])
#         self.down_se_blocks = nn.ModuleList([])
#         for _ in range(self.n_levels):
#             # down convolution
#             if len(self.down_conv_layers) == 0:
#                 downconv = DoubleConv(self.n_input_channels, self.n_filters) # (B, 1, H, W) -> (B, 64, H, W)
#             elif len(self.down_conv_layers) == 1:
#                 downconv = DoubleConv(self.n_filters, self.n_filters)
#             else:
#                 # downconv = DoubleConv(self.n_filters * 2, self.n_filters)
#                 downconv = DoubleConv(self.n_filters, self.n_filters * 2)
#                 self.n_filters *= 2
#             self.down_conv_layers.append(downconv)

            
#             # SE blocks following the downconv 
#             down_se = SE_block(num_features=self.n_filters)  # (B, 64, H, W) -> (B, 64, H, W)
#             self.down_se_blocks.append(down_se)


#         self.middleConv = DoubleConv(
#             self.n_filters, self.n_filters, kernel_size=3, stride=1
#             )
#         self.middleSE = SE_block(num_features = self.n_filters)


#         f = [512+2, 512+256+2, 256+256+2, 128+256+2, 64+256+2, 32+256+2]
#         self.up_conv_layers = nn.ModuleList([])
#         self.up_se_blocks = nn.ModuleList([])
#         for i in range(self.n_levels):
#             # up convolution
#             # if len(self.up_conv_layers) == 0:
#             #     upconv = DoubleConv(self.n_filters+2, self.n_filters//2)
#             # else:
#             #     upconv = DoubleConv(
#             #         # (self.n_filters // 4) * 5 + 2 * self.n_filters+2, self.n_filters
#             #         768+2, self.n_filters
#             #     )
#             # self.n_filters //= 2

#             upconv = DoubleConv(f[i], self.n_filters//2)

#             self.up_conv_layers.append(upconv)

#              # SE blocks following the upconv 
#             up_se = SE_block(num_features=self.n_filters//2)            
#             self.up_se_blocks.append(up_se)


#         for modules in [self.down_conv_layers, self.down_se_blocks,
#                         self.up_conv_layers, self.up_se_blocks,
#                         ]:
#             for l in modules.modules():
#                 if isinstance(l, nn.Conv2d):
#                     torch.nn.init.normal_(l.weight, std=0.01)
#                     if l.bias is not None:
#                         nn.init.constant_(l.bias, 0)


#     @torch.no_grad()
#     def compute_coordinates_linspace(self, x):
#         # linspace is not supported in ONNX
#         h, w = x.size(2), x.size(3)
#         y_loc = torch.linspace(-1, 1, h, device=x.device)
#         x_loc = torch.linspace(-1, 1, w, device=x.device)
#         y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
#         y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
#         x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
#         locations = torch.cat([x_loc, y_loc], 1)
#         return locations.to(x)


#     @torch.no_grad()
#     def compute_coordinates(self, x):
#         h, w = x.size(2), x.size(3)
#         y_loc = -1.0 + 2.0 * torch.arange(h, device=x.device) / (h - 1)
#         x_loc = -1.0 + 2.0 * torch.arange(w, device=x.device) / (w - 1)
#         y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
#         y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
#         x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
#         locations = torch.cat([x_loc, y_loc], 1)
#         return locations.to(x)
        
        

#     # TESTING: add instance and mask branches only to the final layer of the decoder
#     def forward(self, x, idx=None):
#         down_conv_out_tensors = []
#         # down_pool_out_tensors = []
        
#         # go down
#         for i in range(self.n_levels):
#             # print(x.shape)
#             x = self.down_conv_layers[i](x) # (B, 1, H, W) -> (B, 64, H, W) -> (B, 64, H, W) -> (B, 128, H, W) -> (B, 256, H, W) -> (B, 512, H, W)
#             x = self.down_se_blocks[i](x)   # (B, D, H, W) -> (B, D, H, W)
#             down_conv_out_tensors.append(x) #                 [64/2/2,         64/4/4,          128/8/8,          256/16/16,        512/32/32]
#             x = nn.MaxPool2d(2)(x)
#             # down_pool_out_tensors.append(x)

#             # Skip connection if required
#             # if i > 0:
#             #     x = nn.MaxPool2d(2)(down_pool_out_tensors[-2])
#             #     x = torch.cat([x, down_pool_out_tensors[-1]], dim=1)
#             # print("down", x.shape)
                
#         # middle
#         x = self.middleConv(x)
#         x = self.middleSE(x)
#         # print("middle", x.shape)
        
#         # go up
#         def go_up(x):
#             for i in range(self.n_levels):

#                 # print("up start", x.shape)
#                 # if self.coord_conv:
#                 coord_features = self.compute_coordinates(x)
#                 x = torch.cat([coord_features, x], dim=1)
                
#                 x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
#                 # print(self.up_conv_layers[i])
#                 x = self.up_conv_layers[i](x)
#                 # print("before se ->", x.shape)
#                 x = self.up_se_blocks[i](x)
#                 # print("after se ->", x.shape)

#                 x = torch.cat([x, down_conv_out_tensors[-(i + 1)]], dim=1)
#                 # for m in down_conv_out_tensors:
#                 #     print(m.shape)
#                 # print("up end", x.shape)
                
                
#                 # multi-level
#                 # if self.multi_level:
#                 # if i != 0:
#                 #     mb = nn.UpsamplingBilinear2d(scale_factor=2)(mb)    # (1, 128, 128, 128)
#                 #     mb = torch.cat([x, mb], dim=1)
#                 #     mb = self.mask_branch[i](mb)     
#                 # else:
#                 #     mb = self.mask_branch[i](x)

#                 # if i != 0:
#                 #     # scale: (B, N, Hx, Wx) -> (B, N, Hx * 2, Wx * 2)
#                 #     # x features shape: (B, Di, Hx * 2, Wx * 2)
#                 #     inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
#                 #     inst_feats = torch.cat([x, inst_feats], dim=1)
#                 #     inst_feats = self.prior_instance_branch[i](inst_feats)
#                 # else:
#                 #     # inst_feats shape: (B, Dm, Hx, Wx)
#                 #     inst_feats = self.prior_instance_branch[i](x)
                    
#                 # single-level
#                 # else:
#                 if i == self.n_levels - 1:
#                     mb = self.mask_branch[0](x)
#                     inst_feats = self.prior_instance_branch[0](x)
                        
#                 if i == self.n_levels - 1:
#                     logits, kernel, scores, iam = self.instance_branch(inst_feats, idx)

#             return x, mb, (logits, kernel, scores, iam)
    
#         # cyto
#         x, mask_features, (logits, kernel, scores, iam) = go_up(x)
        
#         # Predicting instance masks
#         N = kernel.shape[1]  # num_masks
#         B, C, H, W = mask_features.shape

#         masks = torch.bmm(
#             kernel,    # (B, N, 128)
#             mask_features.view(B, C, H * W)   # (B, 128, [HW])
#         ) # -> (B, N, [HW])
#         masks = masks.view(B, N, H, W)  # (B, N, [HW]) -> (B, N, H, W)
        
#         output = {
#             'pred_logits': logits,
#             'pred_scores': scores,
#             'pred_iam': iam,
#             'pred_masks': masks,
#             'pred_kernel': kernel,
#         }
    
#         return output


from collections import OrderedDict
class SparseSEUnet(nn.Module):

    def __init__(self, cfg: cfg):
        super(SparseSEUnet, self).__init__()
        init_features = 32

        self.cfg = cfg
        self.n_input_channels = cfg.model.in_channels
        self.n_output_channels = cfg.model.out_channels
        self.n_levels = cfg.model.n_levels

        self.coord_conv = cfg.model.coord_conv
        self.multi_level = cfg.model.multi_level
        self.kernel_dim = cfg.model.kernel_dim
        self.num_masks = cfg.model.num_masks
        self.num_convs = cfg.model.num_convs

        features = init_features
        self.encoder1 = SparseSEUnet._block(self.n_input_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = SparseSEUnet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = SparseSEUnet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = SparseSEUnet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = SparseSEUnet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            512, 512, kernel_size=2, stride=2
        )
        self.decoder4 = SparseSEUnet._block(768, 512, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            512, 512, kernel_size=2, stride=2
        )
        self.decoder3 = SparseSEUnet._block(640, 512, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            512, 512, kernel_size=2, stride=2
        )
        self.decoder2 = SparseSEUnet._block(576, 512, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            512, 512, kernel_size=2, stride=2
        )
        self.decoder1 = SparseSEUnet._block(544, 512, name="dec1")



        self.mask_branch = MaskBranch(512)
        
        # instance features.
        self.prior_instance_branch = PriorInstanceBranch(in_channels=512, out_channels=256, num_convs=4)
        
        # instance branch.
        self.instance_branch = InstanceBranch(dim=256, kernel_dim=128, num_masks=25)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        x = self.decoder1(dec1)
        
        mask_features = self.mask_branch(x)
        inst_feats = self.prior_instance_branch(x)
            
        logits, kernel, scores, iam = self.instance_branch(inst_feats)

        
        
        # Predicting instance masks
        N = kernel.shape[1]  # num_masks
        B, C, H, W = mask_features.shape

        masks = torch.bmm(
            kernel,    # (B, N, 128)
            mask_features.view(B, C, H * W)   # (B, 128, [HW])
        ) # -> (B, N, [HW])
        masks = masks.view(B, N, H, W)  # (B, N, [HW]) -> (B, N, H, W)
        
        output = {
            'pred_logits': logits,
            'pred_scores': scores,
            'pred_iam': iam,
            'pred_masks': masks,
            'pred_kernel': kernel,
        }
    
        return output

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


if __name__ == "__main__":
    model = SparseSEUnet(cfg)
    x = torch.rand(1, 1, 512, 512)
    out = model(x)
    print(out["pred_masks"].shape)
