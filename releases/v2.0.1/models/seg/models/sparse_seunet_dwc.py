import torch
from torch import nn
from torch.nn import functional as F

import sys
sys.path.append("./")

from models.seg.heads.instance_head import InstanceBranch, PriorInstanceBranch
from models.seg.heads.mask_head import MaskBranch

from models.seg.models.base import SparseSEUnet as BaseModel
from models.seg.models.base import DoubleConv, SE_block, PyramidPooling
from models.seg.heads.common import Block, FusionConv, DWCFusion, CMUNeXtBlock
from configs import cfg

from utils.registry import MODELS


@MODELS.register(name="sparse_seunet_dwc")
class SparseSEUnet(BaseModel):
    def __init__(
        self,
        cfg: cfg,
        n_filters=64,
        pyramid_pooling=True,
        n_pp_features=144,
    ):
        super(SparseSEUnet, self).__init__(cfg, n_filters, pyramid_pooling, n_pp_features)  
        
        self.coord_conv = cfg.model.coord_conv
        self.multi_level = cfg.model.multi_level
        self.kernel_dim = cfg.model.kernel_dim
        self.num_masks = cfg.model.num_masks
        self.num_convs = cfg.model.num_convs


        self.up_skip_fusion_layers = nn.ModuleList([])
        for _ in range(self.n_levels):
            # up convolution
            if len(self.up_skip_fusion_layers) == 0:
                upconv = nn.Sequential(
                    # nn.Conv2d(208, 208, kernel_size=3, padding=1),
                    # nn.BatchNorm2d(208),
                    # nn.ReLU(inplace=True),
                    # Block(208),
                    # DWCFusion(128, 128)
                    FusionConv(208, 208)
                    )
            else:
                upconv = nn.Sequential(
                    # nn.Conv2d(208, 208, kernel_size=3, padding=1),
                    # nn.BatchNorm2d(208),
                    # nn.ReLU(inplace=True),
                    # Block(208),
                    # DWCFusion(128, 128)
                    FusionConv(208, 208)
                    )
            self.up_skip_fusion_layers.append(upconv)

        depths=[1, 1, 1, 3, 1]
        kernels=[3, 3, 7, 7, 7]
        self.down_conv_layers = nn.ModuleList([])
        for i in range(self.n_levels):
            # down convolution
            if len(self.down_conv_layers) == 0:
                downconv = DoubleConv(self.n_input_channels, self.n_filters)
            elif len(self.down_conv_layers) == 1:
                downconv = nn.Sequential(
                    # Block(self.n_filters), 
                    DoubleConv(self.n_filters, self.n_filters)
                    # CMUNeXtBlock(self.n_filters, self.n_filters, depth=depths[i], k=kernels[i])
                    )
            else:
                downconv = nn.Sequential(
                    # Block(self.n_filters * 2), 
                    DoubleConv(self.n_filters * 2, self.n_filters)
                    # CMUNeXtBlock(self.n_filters * 2, self.n_filters, depth=depths[i], k=kernels[i])
                    )
            self.down_conv_layers.append(downconv)


        self.up_conv_layers = nn.ModuleList([])
        # CHANGED: removed additional +2 channels
        # CHANGED: moved the coordinate features addition to instance branches
        for _ in range(self.n_levels):
            # up convolutions
            if len(self.up_conv_layers) == 0:
                upconv = DoubleConv(self.n_filters, self.n_filters)
            else:
                upconv = DoubleConv(
                    (self.n_filters // 4) * 5 + 2 * self.n_filters, self.n_filters
                )
            self.up_conv_layers.append(upconv)

        
        # mask branch.
        self.mask_branch = nn.ModuleList([])
        self.mask_branch.append(MaskBranch(in_channels=208*5+2, out_channels=256, num_convs=4))
        # for i in range(self.n_levels):
        #     if i == 0:
        #         self.mask_branch.append(MaskBranch(208, num_convs=self.num_convs))
        #     else:
        #         self.mask_branch.append(MaskBranch(208*self.n_levels, num_convs=self.num_convs))
        
        # instance features.
        self.prior_instance_branch = nn.ModuleList([])
        self.prior_instance_branch.append(PriorInstanceBranch(in_channels=208*5+2, out_channels=256, num_convs=4))
        # for i in range(self.n_levels):
        #     if i == 0:
        #         self.prior_instance_branch.append(PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=self.num_convs))
        #     else:
        #         self.prior_instance_branch.append(PriorInstanceBranch(in_channels=208*self.n_levels, out_channels=256, num_convs=self.num_convs))

        # instance branch.
        self.instance_branch = InstanceBranch(dim=256+2, kernel_dim=self.kernel_dim, num_masks=self.num_masks)

        self.projection = nn.Conv2d(256, self.kernel_dim, kernel_size=1)
        
        

    # TESTING: add instance and mask branches only to the final layer of the decoder
    def forward(self, x, idx=None):
        down_conv_out_tensors = []
        down_pp_out_tensors = []
        down_pool_out_tensors = []
        
        # go down
        for i in range(self.n_levels):
            x = self.down_conv_layers[i](x)
            x = self.down_se_blocks[i](x)
            down_conv_out_tensors.append(x)

            # print(f"x: {x.shape}, skip: {down_conv_out_tensors[i].shape}")

            if self.pyramid_pooling:
                x_pp = self.down_pp_layers[i](x)
                x_pp = self.pp_se_blocks[i](x_pp)
                down_pp_out_tensors.append(x_pp)

            x = nn.MaxPool2d(2)(x)
            down_pool_out_tensors.append(x)

            # print("-" * 20)
            # print(f"before skip: {x.shape}")
            # Skip connection if required
            if i > 0:
                x = nn.MaxPool2d(2)(down_pool_out_tensors[-2])
                x = torch.cat([x, down_pool_out_tensors[-1]], dim=1)
            # print(f"after skip: {x.shape}")
            # print(f"before skip: {x.shape}")
                
        # middle
        x = self.middleConv(x)
        x = self.middleSE(x)

        # print(f"middle: {x.shape}")
        # print("=" * 20)
        # print()
        # print("=" * 20)
        
        # go up
        feats = []
        def go_up(x):
            for i in range(self.n_levels):
                # if self.coord_conv:
                # coord_features = self.compute_coordinates(x)
                # x = torch.cat([coord_features, x], dim=1)

                # print(f"before upsample: {x.shape}")
                
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
                x = self.up_conv_layers[i](x)
                x = self.up_se_blocks[i](x)
                
                # print(f"after upsample: {x.shape}")
                # print("-" * 20)
                # print(f"x: {x.shape}, skip: {down_conv_out_tensors[-(i + 1)].shape}")
                # print("-" * 20)

                if self.pyramid_pooling:
                    x = torch.cat([x, down_pp_out_tensors[-(i + 1)]], dim=1)
                else:
                    x = torch.cat([x, down_conv_out_tensors[-(i + 1)]], dim=1)

                feats.append(F.interpolate(x, size=[512, 512], mode='bilinear', align_corners=False))
                
                # print(f"before fusion: {x.shape}")
                # print("-" * 20)

                # x = self.up_skip_fusion_layers[i](x)

                # print(f"after fusion: {x.shape}")
                # print("=" * 20)
                
                
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
                #     x = torch.cat([coord_features, x], dim=1)

                #     mb = self.mask_branch[i](x)

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
                #     x = torch.cat([coord_features, x], dim=1)

                #     inst_feats = self.prior_instance_branch[i](x)
                    
                # single-level
                # else:
                #     if i == self.n_levels - 1:
                #         x = torch.cat(feats, dim=1)
                #         mb = self.mask_branch[0](x)
                #         inst_feats = self.prior_instance_branch[0](x)
                        
                # if i == self.n_levels - 1:
                #     mb = self.projection(mb)

                #     coord_features = self.compute_coordinates(inst_feats)
                #     inst_feats = torch.cat([coord_features, inst_feats], dim=1)

                #     logits, kernel, scores, iam = self.instance_branch(inst_feats, idx)
            
            x = torch.cat(feats, dim=1)

            coord_features = self.compute_coordinates(x)
            x = torch.cat([coord_features, x], dim=1)

            mb = self.mask_branch[0](x)
            mb = self.projection(mb)
            inst_feats = self.prior_instance_branch[0](x)
            
            coord_features = self.compute_coordinates(inst_feats)
            inst_feats = torch.cat([coord_features, inst_feats], dim=1)

            logits, kernel, scores, iam = self.instance_branch(inst_feats, idx)

            return x, mb, (logits, kernel, scores, iam)
    
        # cyto
        x, mask_features, (logits, kernel, scores, iam) = go_up(x)
        
        # Predicting instance masks
        _, N, D = kernel.shape 
        B, C, H, W = mask_features.shape

        # masks = torch.bmm(
        #     kernel,    # (B, N, 128)
        #     mask_features.view(B, C, H * W)   # (B, 128, [HW])
        # ) # -> (B, N, [HW])
        # masks = masks.view(B, N, H, W)  # (B, N, [HW]) -> (B, N, H, W)

        masks = []
        for b in range(len(kernel)):
            m = mask_features[b].unsqueeze(0)
            k = kernel[b]
            k = k.view(N, D, 1, 1)

            inst = F.conv2d(m, k, stride=1)
            masks.append(inst)
        masks = torch.cat(masks, dim=0)
        
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
