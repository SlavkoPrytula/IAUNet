# copied from 45966022
import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn.weight_init import c2_msra_fill

import sys
sys.path.append("./")

from models.seg.heads.instance_head import PriorInstanceBranch, InstanceBranch, DilatedInstanceBranch
from models.seg.heads.mask_head import MaskBranch

from models.seg.models.base import DoubleConv, SE_block
from models.seg.nn.blocks import (PyramidPooling, PyramidPooling_v3, PyramidPooling_v5, 
                             DoubleConv_v2, DoubleConv_v3, DoubleConvModule)


# from ...heads.instance_head import InstanceBranch, PriorInstanceBranch
# from ...heads.mask_head import MaskBranch

# from models.seg.models.base import DoubleConv, SE_block
# from ...blocks.tests import (PyramidPooling, PyramidPooling_v3, PyramidPooling_v5, 
#                              DoubleConv_v2, DoubleConv_v3, DoubleConvModule)
# from ...heads.common import FusionConv, CMUNeXtBlock

from configs import cfg
from utils.registry import MODELS, HEADS




@MODELS.register(name="iaunet_optim_occluder_v3")
class IAUNet(nn.Module):
    def __init__(
        self,
        cfg: cfg,
        # embed_dims=[64, 128, 256, 512, 1024],
        embed_dims=[32, 96, 192, 384, 768],
        # embed_dims=[32, 64, 128, 256, 512],
        # embed_dims=[64, 64, 64, 64, 64],
        pyramid_pooling=True,
        n_pp_features=128,
    ):
        super().__init__()
        
        self.coord_conv = cfg.model.coord_conv
        self.multi_level = cfg.model.multi_level
        self.kernel_dim = cfg.model.instance_head.kernel_dim
        self.num_convs = cfg.model.num_convs

        self.cfg = cfg  
        self.n_input_channels = cfg.model.in_channels
        self.n_output_channels = cfg.model.out_channels
        self.n_levels = cfg.model.n_levels

        self.embed_dims = embed_dims
        self.n_pp_features = n_pp_features
        self.pyramid_pooling = pyramid_pooling
        self.kernel_strides_map = {1: 16, 2: 8, 3: 4, 4: 2, 5: 1}

        self.skips = True

        self.down_conv_layers = nn.ModuleList([])
        self.down_pp_layers = nn.ModuleList([])
        self.down_se_blocks = nn.ModuleList([])
        self.up_se_blocks = nn.ModuleList([])
        self.pp_se_blocks = nn.ModuleList([])
        self.up_conv_layers = nn.ModuleList([])
        self.up_conv_layers = nn.ModuleList([])

        self.middleConv = DoubleConvModule(self.embed_dims[-1] + self.embed_dims[-2], self.embed_dims[-1])
        self.middleSE = SE_block(num_features = self.embed_dims[-1])

        depths = [1, 1, 1, 3, 1] #m
        # depths = [1, 1, 3, 6, 2]
        for i in range(self.n_levels):
            # down convolution
            if i == 0:
                in_channels = self.n_input_channels
            elif i == 1:
                in_channels = embed_dims[i-1]
            else:
                in_channels = embed_dims[i-1] + embed_dims[i-2]

            downconv = DoubleConvModule(in_channels, self.embed_dims[i], depth=depths[i])
            # downconv = DoubleConv(in_channels, self.embed_dims[i])
            self.down_conv_layers.append(downconv)
           
            # SE blocks following the downconv 
            down_se = SE_block(num_features=self.embed_dims[i])
            self.down_se_blocks.append(down_se)


            # down pyramid
            if self.pyramid_pooling:
                pplayer = PyramidPooling_v5(in_channels=self.embed_dims[i], 
                                            pool_sizes=[1, 2, 4, 8], 
                                            out_channels=self.n_pp_features, 
                                            expand=1)
                self.down_pp_layers.append(pplayer)

                # SE blocks following the pp block
                pp_se = SE_block(num_features=self.n_pp_features)            
                self.pp_se_blocks.append(pp_se)
            
        
        embed_dims = self.embed_dims[::-1]
        for i in range(self.n_levels):
            if i == 0:
                in_channels = embed_dims[i] + 2
            else:
                in_channels = embed_dims[i-1] + self.n_pp_features + 2

            upconv = DoubleConv_v2(in_channels, embed_dims[i])
            # upconv = FusionConv(in_channels, embed_dims[i])
            self.up_conv_layers.append(upconv)

            up_se = SE_block(num_features=embed_dims[i])
            self.up_se_blocks.append(up_se)


        self.up_skip_layers = nn.ModuleList([])
        for i in range(self.n_levels):
            upconv = DoubleConv_v2(embed_dims[i] + self.n_pp_features, embed_dims[i])
            # upconv = FusionConv(in_channels, embed_dims[i])
            self.up_skip_layers.append(upconv)
            
        
        # mask branch.
        mask_dim = cfg.model.mask_dim
        self.mask_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                self.mask_branch.append(MaskBranch(embed_dims[i], out_channels=mask_dim, num_convs=self.num_convs))
            else:
                self.mask_branch.append(MaskBranch(embed_dims[i] + mask_dim, out_channels=mask_dim, num_convs=self.num_convs))


        # self.mask_branch = MaskBranch(dim, out_channels=mask_dim, num_convs=self.num_convs)
        self.projection = nn.Conv2d(mask_dim, self.kernel_dim, kernel_size=1)
        c2_msra_fill(self.projection)
        
        # instance features.
        self.prior_instance_branch = nn.ModuleList([])
        self.prior_occulder_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                self.prior_instance_branch.append(PriorInstanceBranch(in_channels=embed_dims[i], out_channels=mask_dim, num_convs=self.num_convs))
                self.prior_occulder_branch.append(PriorInstanceBranch(in_channels=embed_dims[i], out_channels=mask_dim, num_convs=self.num_convs))
            else:
                self.prior_instance_branch.append(PriorInstanceBranch(in_channels=embed_dims[i] + mask_dim + 2, out_channels=mask_dim, num_convs=self.num_convs))
                self.prior_occulder_branch.append(PriorInstanceBranch(in_channels=embed_dims[i] + mask_dim + 2, out_channels=mask_dim, num_convs=self.num_convs))

        # instance branch.
        self.instance_head = HEADS.build(cfg.model.instance_head)
        self.occulder_head = HEADS.build(cfg.model.instance_head)


        for modules in [self.down_conv_layers, self.down_se_blocks,
                        self.up_conv_layers, self.up_se_blocks,
                        self.down_pp_layers, self.pp_se_blocks]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)


    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        x_loc = torch.linspace(-1, 1, h, device=x.device)
        y_loc = torch.linspace(-1, 1, w, device=x.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x_loc, y_loc], 1)

        return coord_feat
        

    # TESTING: add instance and mask branches only to the final layer of the decoder
    def forward(self, x, idx=None):
        down_pp_out_tensors = []
        down_pool_out_tensors = []
        
        # go down
        for i in range(self.n_levels):
            x = self.down_conv_layers[i](x)
            x = self.down_se_blocks[i](x)
            
            if self.pyramid_pooling:
                x_pp = self.down_pp_layers[i](x)
                x_pp = self.pp_se_blocks[i](x_pp)
                down_pp_out_tensors.append(x_pp)
            x = nn.MaxPool2d(2)(x)
            
            down_pool_out_tensors.append(x)
            # residual connection if required
            if i > 0:
                x = nn.MaxPool2d(2)(down_pool_out_tensors[-2])
                x = torch.cat([x, down_pool_out_tensors[-1]], dim=1)


        # middle
        x = self.middleConv(x)
        x = self.middleSE(x)


        # go up
        # def go_up(x):
        for i in range(self.n_levels):
            # if self.coord_conv:
            x = nn.UpsamplingBilinear2d(scale_factor=2)(x)

            coord_features = self.compute_coordinates(x)
            x = torch.cat([coord_features, x], dim=1)
            
            x = self.up_conv_layers[i](x)
            x = self.up_se_blocks[i](x)
            
            if self.pyramid_pooling:
                x = torch.cat([x, down_pp_out_tensors[-(i + 1)]], dim=1)
                skip_x = self.up_skip_layers[i](x)

            
            # multi-level
            # if self.multi_level:
            if i != 0:
                mb = nn.UpsamplingBilinear2d(scale_factor=2)(mb)    # (1, 128, 128, 128)
                mb = torch.cat([skip_x, mb], dim=1)
                mb = self.mask_branch[i](mb)     
            else:
                mb = self.mask_branch[i](skip_x)

            if i != 0:
                # scale: (B, N, Hx, Wx) -> (B, N, Hx * 2, Wx * 2)
                # x features shape: (B, Di, Hx * 2, Wx * 2)
                inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
                
                inst_feats = torch.cat([skip_x, inst_feats], dim=1)
                coord_features = self.compute_coordinates(inst_feats)

                inst_feats = torch.cat([coord_features, inst_feats], dim=1)
                inst_feats = self.prior_instance_branch[i](inst_feats)
            
            elif i == self.n_levels - 1:
                inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
                occl_feats = inst_feats.clone()
                
                inst_feats = torch.cat([skip_x, inst_feats], dim=1)
                coord_features = self.compute_coordinates(inst_feats)
                inst_feats = torch.cat([coord_features, inst_feats], dim=1)
                inst_feats = self.prior_instance_branch[i](inst_feats)

                occl_feats = torch.cat([skip_x, occl_feats], dim=1)
                coord_features = self.compute_coordinates(occl_feats)
                occl_feats = torch.cat([coord_features, occl_feats], dim=1)
                occl_feats = self.prior_occulder_branch[i](occl_feats)
            
            else:
                # inst_feats shape: (B, Dm, Hx, Wx)
                inst_feats = self.prior_instance_branch[i](skip_x)
                    
        logits, kernel, scores, iam = self.instance_head(inst_feats, idx)
        occluder_logits, occluder_kernel, occluder_scores, occluder_iam = self.occulder_head(inst_feats, idx)
        mb = self.projection(mb)
        
        # Predicting instance masks
        N = kernel.shape[1]
        B, C, H, W = mb.shape

        masks = torch.bmm(
            kernel, mb.view(B, C, H * W)
        )
        masks = masks.view(B, N, H, W)

        occluder_masks = torch.bmm(
            occluder_kernel, mb.view(B, C, H * W)
        )
        occluder_masks = occluder_masks.view(B, N, H, W)


        output = {
            'pred_logits': logits,
            'pred_scores': scores,
            'pred_iam': iam,
            'pred_masks': masks,
            
            'pred_occluders_logits': occluder_logits,
            'pred_occluders_scores': occluder_scores,
            'pred_occluders_iam': occluder_iam,
            'pred_occluders_masks': occluder_masks,
        }
    
        return output
    

if __name__ == "__main__":
    import time 

    model = IAUNet(cfg)
    x = torch.rand(1, 3, 512, 512)
    
    time_s = time.time()
    out = model(x)
    time_e = time.time()
    print(f'done in {time_e - time_s}(s)')