import torch
from torch import nn
import numpy as np
from fvcore.nn.weight_init import c2_msra_fill

import sys
sys.path.append("./")

from models.seg.heads.instance_head import InstanceBranch, PriorInstanceBranch
from models.seg.heads.mask_head import MaskBranch

from models.seg.models.base import SparseSEUnet as BaseModel
from models.seg.models.base import DoubleConv, SE_block, PyramidPooling
from models.seg.heads.common import Block, FusionConv, DWCFusion
from configs import cfg

from utils.registry import MODELS


@MODELS.register(name="iaunet")
class IAUNet(BaseModel):
    def __init__(
        self,
        cfg: cfg,
        n_filters=64,
        dims=[64, 128, 256, 512, 1024],
        drp=[],
        pyramid_pooling=False,
        n_pp_features=144,
    ):
        super(IAUNet, self).__init__(cfg, n_filters, pyramid_pooling, n_pp_features)  
        
        self.coord_conv = cfg.model.coord_conv
        self.multi_level = cfg.model.multi_level
        self.kernel_dim = cfg.model.kernel_dim
        self.num_masks = cfg.model.num_masks
        self.num_convs = cfg.model.num_convs

        self.debug = False
        self.embed_dims = dims

        
        self.down_conv_layers = nn.ModuleList([])
        self.down_se_blocks = nn.ModuleList([])
        for i in range(self.n_levels):
            # down convolution
            if len(self.down_conv_layers) == 0:
                downconv = DoubleConv(self.n_input_channels, dims[i])
            else:
                downconv = DoubleConv(dims[i-1], dims[i])
            self.down_conv_layers.append(downconv)

            # SE blocks following the downconv 
            down_se = SE_block(num_features=dims[i])
            self.down_se_blocks.append(down_se)


        self.middleConv = DoubleConv(dims[-1], dims[-1], kernel_size=3, stride=1)
        self.middleSE = SE_block(num_features=dims[-1])

        dims = dims[::-1]


        # self.embed_dims = np.cumsum(self.embed_dims)[::-1]
        self.up_conv_layers = nn.ModuleList([])
        self.up_se_blocks = nn.ModuleList([])
        for i in range(self.n_levels):
            # up convolution
            if i == 0:
                upconv = DoubleConv(dims[i] + 2, dims[i])
                print(dims[i] + 2, "->", dims[i])
            else:
                upconv = DoubleConv(dims[i-1] + 2, dims[i])
                print(dims[i-1] + 2, "->", dims[i])

            self.up_conv_layers.append(upconv)

             # SE blocks following the upconv 
            up_se = SE_block(num_features=dims[i])  
            self.up_se_blocks.append(up_se)


        self.up_skip_fusion_layers = nn.ModuleList([])
        for i in range(self.n_levels):
            # up convolution
            if i != self.n_levels - 1:
                upconv = nn.Sequential(
                    FusionConv(dims[i]*2, dims[i])
                    )
            else:
                upconv = nn.Sequential(
                    FusionConv(dims[i]*2, dims[i]*2)
                    )
            self.up_skip_fusion_layers.append(upconv)


        # mask branch.
        mask_dim = self.cfg.model.mask_dim
        self.mask_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                self.mask_branch.append(
                    MaskBranch(dims[i], out_channels=mask_dim, num_convs=self.num_convs)
                    )
            elif i < self.n_levels - 1:
                self.mask_branch.append(
                    MaskBranch(dims[i] + mask_dim, out_channels=mask_dim, num_convs=self.num_convs)
                    )
            else:
                self.mask_branch.append(
                    MaskBranch(dims[i]*2 + mask_dim, out_channels=mask_dim, num_convs=self.num_convs)
                    )

        self.projection = nn.Conv2d(mask_dim, self.kernel_dim, kernel_size=1)
        c2_msra_fill(self.projection)
        
        # instance features.
        self.prior_instance_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                self.prior_instance_branch.append(
                    PriorInstanceBranch(in_channels=dims[i], 
                                        out_channels=mask_dim, 
                                        num_convs=self.num_convs)
                    )
            elif i < self.n_levels - 1:
                self.prior_instance_branch.append(
                    PriorInstanceBranch(in_channels=dims[i] + mask_dim, 
                                        out_channels=mask_dim, 
                                        num_convs=self.num_convs)
                    )
            else: 
                self.prior_instance_branch.append(
                    PriorInstanceBranch(in_channels=dims[i]*2 + mask_dim, 
                                        out_channels=mask_dim, 
                                        num_convs=self.num_convs)
                    )

        # instance branch.
        self.instance_branch = InstanceBranch(dim=mask_dim, 
                                              kernel_dim=self.kernel_dim, 
                                              num_masks=self.num_masks)
        
        
    def forward(self, x, idx=None):
        down_conv_out_tensors = []
        # down_pp_out_tensors = []
        # down_pool_out_tensors = []

        # down            +  skips:
        # (3, 512, 512)   -> (64, 512, 512)
        # (64, 256, 256)
        # (64, 256, 256)  -> (128, 256, 256)
        # (128, 128, 128)
        # (128, 128, 128) -> (256, 128, 128)
        # (256, 64, 64)
        # (256, 64, 64)   -> (512, 64, 64)
        # (512, 32, 32)
        # (512, 32, 32)   -> (1024, 32, 32)
        # (1024, 16, 16)

        # middle conv:
        # (1024, 16, 16)  -> (1024, 16, 16)

        # up:
        # (1024, 16, 16)  -> (1024, 32, 32)
        # (1024, 32, 32) + (1024, 32, 32) -> (2048, 32, 32) -> (1024, 32, 32)
        # (1024, 32, 32) -> (1024, 64, 64) -> (512, 64, 64)
        # (512, 64, 64) + (512, 64, 64) -> (1024, 64, 64) -> (512, 64, 64)
        
        # go down
        for i in range(self.n_levels):
            x = self.down_conv_layers[i](x)
            x = self.down_se_blocks[i](x)
            down_conv_out_tensors.append(x)

            x = nn.MaxPool2d(2)(x)
            
        # middle
        x = self.middleConv(x)
        x = self.middleSE(x)

        # go up
        def go_up(x):
            for i in range(self.n_levels):
                coord_features = self.compute_coordinates(x)
                x = torch.cat([coord_features, x], dim=1)
                # print(x.shape)

                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
                x = self.up_conv_layers[i](x)
                x = self.up_se_blocks[i](x)

                # skip 
                # print(i)
                # print(x.shape, down_conv_out_tensors[-(i + 1)].shape)
                x = torch.cat([x, down_conv_out_tensors[-(i + 1)]], dim=1)
                # x = torch.cat([x, down_conv_out_tensors[-i]], dim=1)   # adjust for removing the last skip
                # print(x.shape)
                x = self.up_skip_fusion_layers[i](x)
                # print(x.shape)
   
                if i != 0:
                    inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
                    inst_feats = torch.cat([x, inst_feats], dim=1)
                    inst_feats = self.prior_instance_branch[i](inst_feats)

                    mb = nn.UpsamplingBilinear2d(scale_factor=2)(mb)
                    mb = torch.cat([x, mb], dim=1)
                    mb = self.mask_branch[i](mb)    

                if i == 0:
                    mb = self.mask_branch[i](x)
                    inst_feats = self.prior_instance_branch[i](x)
                    
                # if i == self.n_levels - 1:

            # print(inst_feats.shape)
            # print(mb.shape)
            logits, kernel, scores, iam = self.instance_branch(inst_feats, idx)
            mb = self.projection(mb)

            return x, mb, (logits, kernel, scores, iam)
    
        x, mask_features, (logits, kernel, scores, iam) = go_up(x)
        
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



if __name__ == "__main__":
    model = SparseSEUnet(cfg)
    x = torch.rand(1, 3, 512, 512)
    out = model(x)
    print(out["pred_masks"].shape)
