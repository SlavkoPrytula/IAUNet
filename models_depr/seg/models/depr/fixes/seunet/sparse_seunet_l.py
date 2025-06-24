import torch
from torch import nn
import numpy as np

import sys
sys.path.append("./")

from models.seg.heads.instance_head import InstanceBranch, PriorInstanceBranch
from models.seg.heads.mask_head import MaskBranch

from models.seg.models.base import SparseSEUnet as BaseModel
from models.seg.models.base import DoubleConv, SE_block, PyramidPooling
from models.seg.heads.common import Block, FusionConv, DWCFusion
from configs import cfg

from utils.registry import MODELS


@MODELS.register(name="sparse_seunet_l")
class SparseSEUnet(BaseModel):
    def __init__(
        self,
        cfg: cfg,
        n_filters=64,
        dims=[64, 256, 256, 256, 256],
        # dims=[64, 64, 64, 64, 64],
        drp=[],
        pyramid_pooling=False,
        n_pp_features=144,
    ):
        super(SparseSEUnet, self).__init__(cfg, n_filters, pyramid_pooling, n_pp_features)  
        
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
                downconv = DoubleConv(self.n_input_channels, self.embed_dims[i])
            else:
                downconv = DoubleConv(self.embed_dims[i-1], self.embed_dims[i])
            self.down_conv_layers.append(downconv)

            # SE blocks following the downconv 
            down_se = SE_block(num_features=self.embed_dims[i])
            self.down_se_blocks.append(down_se)


        self.middleConv = DoubleConv(self.embed_dims[-1], self.embed_dims[-1], kernel_size=3, stride=1)
        self.middleSE = SE_block(num_features=self.embed_dims[-1])


        # self.embed_dims = np.cumsum(self.embed_dims)[::-1]
        self.up_conv_layers = nn.ModuleList([])
        self.up_se_blocks = nn.ModuleList([])
        for i in range(self.n_levels):
            # up convolution
            upconv = DoubleConv(self.embed_dims[-(i+1)], self.embed_dims[-(i+1)])
            self.up_conv_layers.append(upconv)

             # SE blocks following the upconv 
            up_se = SE_block(num_features=self.embed_dims[-(i+1)])  
            self.up_se_blocks.append(up_se)


        self.up_skip_fusion_layers = nn.ModuleList([])
        for i in range(self.n_levels):
            # up convolution
            if i != self.n_levels - 1:
                upconv = nn.Sequential(
                    FusionConv(self.embed_dims[-(i+1)]*2, self.embed_dims[-(i+1)-1])
                    )
            else:
                upconv = nn.Sequential(
                    FusionConv(self.embed_dims[-(i+1)]*2, self.embed_dims[-(i+1)])
                    )
            self.up_skip_fusion_layers.append(upconv)

        # mask branch.
        self.mask_branch = nn.ModuleList([])
        self.mask_branch.append(MaskBranch(in_channels=self.embed_dims[0]+2, out_channels=256, num_convs=4))
        # for i in range(self.n_levels):
        #     if i == 0:
        #         self.mask_branch.append(MaskBranch(in_channels=self.embed_dims[-(i+1)]+2, out_channels=256, num_convs=self.num_convs))
        #     elif i != self.n_levels - 1:
        #         self.mask_branch.append(MaskBranch(in_channels=self.embed_dims[-(i+1)]+2, out_channels=256, num_convs=self.num_convs))
        #     else:
        #         self.mask_branch.append(MaskBranch(in_channels=self.embed_dims[-(i+1)] + self.embed_dims[-1]+2, num_convs=self.num_convs))
        
        # instance features.
        self.prior_instance_branch = nn.ModuleList([
            PriorInstanceBranch(256+2, out_channels=256, num_convs=self.num_convs),
            PriorInstanceBranch(256+2, out_channels=256, num_convs=self.num_convs),
            PriorInstanceBranch(256+2, out_channels=256, num_convs=self.num_convs),
            PriorInstanceBranch(64+2, out_channels=256, num_convs=self.num_convs),
            PriorInstanceBranch(64+2, out_channels=256, num_convs=self.num_convs)
        ])
        # self.prior_instance_branch.append(PriorInstanceBranch(in_channels=self.embed_dims[0]*2+2, out_channels=256, num_convs=4))
        # for i in range(self.n_levels):
        #     if i != self.n_levels - 1:
        #         self.prior_instance_branch.append(PriorInstanceBranch(self.embed_dims[-(i-1)]+2, out_channels=256, num_convs=self.num_convs))
        #     else:
        #         self.prior_instance_branch.append(PriorInstanceBranch(in_channels=self.embed_dims[-(i+1)] + self.embed_dims[-1]+2, out_channels=256, num_convs=self.num_convs))

        self.iam_blocks = nn.ModuleList([])
        for i in range(self.n_levels):
            self.iam_blocks.append(InstanceBranch(dim=256, kernel_dim=self.kernel_dim, num_masks=self.num_masks))

        # instance branch.
        # self.instance_branch = InstanceBranch(dim=256+2, kernel_dim=self.kernel_dim, num_masks=self.num_masks)

        self.projection = nn.Conv2d(256, self.kernel_dim, kernel_size=1)
        
        

    # TESTING: add instance and mask branches only to the final layer of the decoder
    def forward(self, x, idx=None):
        down_conv_out_tensors = []
        down_pp_out_tensors = []
        down_pool_out_tensors = []
        
        # go down
        for i in range(self.n_levels):
            if self.debug:
                print()
                print(f"[level: {i}]")
                print(f"x in: {x.shape}")
                print("-" * 20)

            x = self.down_conv_layers[i](x)
            x = self.down_se_blocks[i](x)
            down_conv_out_tensors.append(x)

            if self.debug:
                print(f"x: {x.shape}, skip: {down_conv_out_tensors[i].shape}")

            if self.pyramid_pooling:
                x_pp = self.down_pp_layers[i](x)
                x_pp = self.pp_se_blocks[i](x_pp)
                down_pp_out_tensors.append(x_pp)

            x = nn.MaxPool2d(2)(x)
            down_pool_out_tensors.append(x)

            # if self.debug:
            #     print("-" * 20)
            #     print(f"before residual: {x.shape}")
            
            # # # Skip connection if required
            # if i > 0:
            #     x = nn.MaxPool2d(2)(down_pool_out_tensors[-2])
            #     x = torch.cat([x, down_pool_out_tensors[-1]], dim=1)
            #     # x = x + down_pool_out_tensors[-1]
            
            # if self.debug:
            #     print(f"after residual: {x.shape}")
            #     print("-" * 20)
                
        # middle
        x = self.middleConv(x)
        x = self.middleSE(x)
        
        if self.debug:
            print(f"middle: {x.shape}")
            print("=" * 20)
            print()
            print("=" * 20)
        
        # go up
        def go_up(x):
            for i in range(self.n_levels):

                if self.debug:
                    print()
                    print(f"[level: {self.n_levels-i-1}]")
                    print(f"before upsample: {x.shape}")
                
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
                x = self.up_conv_layers[i](x)
                x = self.up_se_blocks[i](x)
                
                if self.debug:
                    print(f"after upsample: {x.shape}")
                    print("-" * 20)
                    print(f"x: {x.shape}, skip: {down_conv_out_tensors[-(i + 1)].shape}")
                    print("-" * 20)

                if self.pyramid_pooling:
                    x = torch.cat([x, down_pp_out_tensors[-(i + 1)]], dim=1)
                else:
                    x = torch.cat([x, down_conv_out_tensors[-(i + 1)]], dim=1)
                
                if self.debug:
                    print(f"before fusion: {x.shape}")
                    print("-" * 20)

                x = self.up_skip_fusion_layers[i](x)
                
                if self.debug:
                    print(f"after fusion: {x.shape}")
                    print("=" * 20)
                
                
                # multi-level
                # if self.multi_level:
                # if i != 0:
                #     mb = nn.UpsamplingBilinear2d(scale_factor=2)(mb)    # (1, 128, 128, 128)
                #     mb = torch.cat([x, mb], dim=1)

                #     coord_features = self.compute_coordinates(mb)
                #     mb = torch.cat([coord_features, mb], dim=1)
                #     mb = self.mask_branch[i](mb)     
                # else:
                #     coord_features = self.compute_coordinates(x)
                #     _x = torch.cat([coord_features, x], dim=1)
                #     mb = self.mask_branch[i](_x)

                coord_features = self.compute_coordinates(x)
                _x = torch.cat([coord_features, x], dim=1)
                
                if i != 0 and i != self.n_levels - 1:
                    iam_l = nn.UpsamplingBilinear2d(scale_factor=2)(iam)
                    inst_feats = self.prior_instance_branch[i](_x)
                    iam = self.iam_blocks[i](inst_feats, return_iam_only=True)
                    iam = iam + iam_l
                else:
                    inst_feats = self.prior_instance_branch[i](_x)
                    iam = self.iam_blocks[i](inst_feats, return_iam_only=True)



                # if i != 0:
                #     # scale: (B, N, Hx, Wx) -> (B, N, Hx * 2, Wx * 2)
                #     # x features shape: (B, Di, Hx * 2, Wx * 2)
                #     inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
                #     inst_feats = torch.cat([x, inst_feats], dim=1)
                    
                #     coord_features = self.compute_coordinates(inst_feats)
                #     inst_feats = torch.cat([coord_features, inst_feats], dim=1)
                #     inst_feats = self.prior_instance_branch[i](inst_feats)
                # else:
                #     # inst_feats shape: (B, Dm, Hx, Wx)
                #     coord_features = self.compute_coordinates(x)
                #     _x = torch.cat([coord_features, x], dim=1)
                #     inst_feats = self.prior_instance_branch[i](_x)
                    
                # # single-level
                # else:
                #     if i == self.n_levels - 1:
                #         mb = self.mask_branch[i](x)
                #         inst_feats = self.prior_instance_branch[i](x)
                        
                if i == self.n_levels - 1:
                    mb = self.mask_branch[0](_x)
                    mb = self.projection(mb)
                    inst_feats = self.prior_instance_branch[i](_x)

                    logits, kernel, scores, iam = self.iam_blocks[i](inst_feats, iam=iam, return_iam_only=False)


            iam = {"instance_iam": iam,}
            return x, mb, (logits, kernel, scores, iam)
    
        # cyto
        x, mask_features, (logits, kernel, scores, iam) = go_up(x)
        
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



if __name__ == "__main__":
    model = SparseSEUnet(cfg)
    x = torch.rand(1, 2, 512, 512)
    out = model(x)
    print(out["pred_masks"].shape)
