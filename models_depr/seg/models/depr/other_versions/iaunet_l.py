# copied from 45966022
import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn.weight_init import c2_msra_fill

import sys
sys.path.append("./")

from models.seg.heads.instance_head import InstanceBranch, PriorInstanceBranch
from models.seg.heads.mask_head import MaskBranch

from models.seg.modules.mixup import MixUpScaler
from models.seg.models.base import SparseSEUnet as BaseModel
from models.seg.models.base import DoubleConv, SE_block, PyramidPooling

from configs import cfg
from utils.registry import MODELS


@MODELS.register(name="iaunet_l")
class IAUNet(BaseModel):
    def __init__(
        self,
        cfg: cfg,
        n_filters=64,
        pyramid_pooling=True,
        n_pp_features=144,
    ):
        super(IAUNet, self).__init__(cfg, n_filters, pyramid_pooling, n_pp_features)  
        
        self.coord_conv = cfg.model.coord_conv
        self.multi_level = cfg.model.multi_level
        self.kernel_dim = cfg.model.kernel_dim
        self.num_masks = cfg.model.num_masks
        self.num_convs = cfg.model.num_convs


        self.down_conv_layers = nn.ModuleList([])
        self.down_se_blocks = nn.ModuleList([])
        self.down_pp_layers = nn.ModuleList([])
        self.pp_se_blocks = nn.ModuleList([])

        self.res_blocks = nn.ModuleList([])
        self.res_se_blocks = nn.ModuleList([])

        for i in range(1, self.n_levels):
            if i == 1:
                res_conv = DoubleConv(self.n_filters * 2, self.n_filters * 2)
                res_se = SE_block(num_features=self.n_filters * 2)
            else:
                res_conv = DoubleConv(self.n_filters * 2 + self.n_filters, self.n_filters * 2)
                res_se = SE_block(num_features=self.n_filters * 2)
            self.res_blocks.append(res_conv)
            self.res_se_blocks.append(res_se)


        for i in range(self.n_levels):
            if i == 0:
                downconv = DoubleConv(self.n_input_channels, self.n_filters)
            elif i == 1:
                downconv = DoubleConv(self.n_filters, self.n_filters)
            else:
                downconv = DoubleConv(self.n_filters * 2, self.n_filters)
            self.down_conv_layers.append(downconv)
            down_se = SE_block(num_features = self.n_filters)
            self.down_se_blocks.append(down_se)

            if self.pyramid_pooling:
                if i == 0:
                    pplayer = PyramidPooling(
                        kernel_strides_map=self.kernel_strides_map, 
                        n_filters=self.n_filters
                    )
                    pp_se = SE_block(num_features=self.n_pp_features)
                else:
                    pplayer = PyramidPooling(
                        kernel_strides_map=self.kernel_strides_map, 
                        n_filters=self.n_filters * 2
                    )
                    pp_se = SE_block(num_features=self.n_pp_features * 2)

                # pplayer = PyramidPooling(
                #     kernel_strides_map=self.kernel_strides_map, 
                #     n_filters=self.n_filters
                #     )
                # pp_se = SE_block(num_features=self.n_pp_features)

                self.down_pp_layers.append(pplayer)
                self.pp_se_blocks.append(pp_se)


        self.middleConv = DoubleConv(self.n_filters * 2, self.n_filters, kernel_size=3, stride=1)
        self.middleSE = SE_block(num_features=self.n_filters)


        # ADDED MANUALLY
        l = [144, 288, 288, 288, 288]
        self.up_conv_layers = nn.ModuleList([])
        for i in range(self.n_levels):
            if len(self.up_conv_layers) == 0:
                upconv = DoubleConv(self.n_filters + 2, self.n_filters)
            else:
                # upconv = DoubleConv((self.n_filters // 4) * 5 + 2 * self.n_filters + 2, self.n_filters)
                upconv = DoubleConv(l[i] + self.n_filters + 2, self.n_filters)
            self.up_conv_layers.append(upconv)
            
        
        # mask branch.
        mask_dim = cfg.model.mask_dim
        self.mask_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                self.mask_branch.append(MaskBranch(352, out_channels=mask_dim, num_convs=self.num_convs))
            elif i < self.n_levels - 1:
                self.mask_branch.append(MaskBranch(352 + mask_dim, out_channels=mask_dim, num_convs=self.num_convs))
            else:
                self.mask_branch.append(MaskBranch(208 + mask_dim, out_channels=mask_dim, num_convs=self.num_convs))

        self.projection = nn.Conv2d(mask_dim, self.kernel_dim, kernel_size=1)
        c2_msra_fill(self.projection)
        
        # instance features.
        self.prior_instance_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                self.prior_instance_branch.append(PriorInstanceBranch(in_channels=352, out_channels=mask_dim, num_convs=self.num_convs))
            elif i < self.n_levels - 1:
                self.prior_instance_branch.append(PriorInstanceBranch(in_channels=352 + mask_dim, out_channels=mask_dim, num_convs=self.num_convs))
            else:
                self.prior_instance_branch.append(PriorInstanceBranch(in_channels=208 + mask_dim, out_channels=mask_dim, num_convs=self.num_convs))

        # instance branch.
        self.instance_branch = InstanceBranch(dim=mask_dim, kernel_dim=self.kernel_dim, num_masks=self.num_masks)
        
        
    def forward(self, x, idx=None):
        down_conv_out_tensors = []
        down_pp_skip = []
        down_residual_feats = []
        
        # go down
        for i in range(self.n_levels):
            # print(x.shape)
            x = self.down_conv_layers[i](x)
            x = self.down_se_blocks[i](x)
            # print(x.shape)

            # residual connection
            if i > 0:
                res = nn.MaxPool2d(2)(down_residual_feats[-1])
                x = torch.cat([x, res], dim=1)
                # print("res_in", x.shape)
                x = self.res_blocks[i-1](x)
                x = self.res_se_blocks[i-1](x)
                # print("res_out", x.shape)

            # prepare skip connections
            down_conv_out_tensors.append(x)
            if self.pyramid_pooling:
                # print(x.shape, i)
                x_pp = self.down_pp_layers[i](x)
                x_pp = self.pp_se_blocks[i](x_pp)
                down_pp_skip.append(x_pp)
            
            down_residual_feats.append(x)
    
            x = nn.MaxPool2d(2)(x)
            
            # # Skip connection if required
            # if i > 0:
            #     x = nn.MaxPool2d(2)(down_residual_feats[-2])
            #     x = torch.cat([x, down_residual_feats[-1]], dim=1)
            #     print(x.shape)
        
        # print("="*25)
        # for a in down_pp_skip:
            # print(a.shape)
                
        # middle
        x = self.middleConv(x)
        x = self.middleSE(x)
        # print("="*25)

        # go up
        for i in range(self.n_levels):
            coord_features = self.compute_coordinates(x)
            x = torch.cat([coord_features, x], dim=1)
            
            # print(x.shape)
            x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
            x = self.up_conv_layers[i](x)
            x = self.up_se_blocks[i](x)
            # print(x.shape)
            
            # print(x.shape)
            if self.pyramid_pooling:
                x = torch.cat([x, down_pp_skip[-(i + 1)]], dim=1)
            else:
                x = torch.cat([x, down_conv_out_tensors[-(i + 1)]], dim=1)
            # print(x.shape)
            
            if i != 0:
                mask_feats = nn.UpsamplingBilinear2d(scale_factor=2)(mask_feats)
                mask_feats = torch.cat([x, mask_feats], dim=1)
                mask_feats = self.mask_branch[i](mask_feats)  

                inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
                inst_feats = torch.cat([x, inst_feats], dim=1)
                inst_feats = self.prior_instance_branch[i](inst_feats)   
            else:
                mask_feats = self.mask_branch[i](x)
                inst_feats = self.prior_instance_branch[i](x)
                    

        logits, kernel, scores, iam = self.instance_branch(inst_feats, idx)
        mask_feats = self.projection(mask_feats)
        
        # Predicting instance masks
        N = kernel.shape[1]  # num_masks
        B, C, H, W = mask_feats.shape

        masks = torch.bmm(
            kernel,    # (B, N, 128)
            mask_feats.view(B, C, H * W)   # (B, 128, [HW])
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
    x = torch.rand(2, 3, 512, 512)
    print(model)
    out = model(x)
    print(out["pred_masks"].shape)