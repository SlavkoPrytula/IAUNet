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
from models.seg.models.base import DoubleConv, SE_block

from configs import cfg
from utils.registry import MODELS


@MODELS.register(name="iaunet_deep_supervision")
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

        # ADDED MANUALLY
        self.up_conv_layers = nn.ModuleList([])
        for _ in range(self.n_levels):
            if len(self.up_conv_layers) == 0:
                upconv = DoubleConv(self.n_filters + 2, self.n_filters)
                # upconv = DoubleConv(self.n_filters, self.n_filters)
            else:
                upconv = DoubleConv(
                    (self.n_filters // 4) * 5 + 2 * self.n_filters + 2, self.n_filters
                )
                # upconv = DoubleConv(
                #     (self.n_filters // 4) * 5 + 2 * self.n_filters, self.n_filters
                # )
            self.up_conv_layers.append(upconv)
            
        
        # mask branch.
        self.mask_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                self.mask_branch.append(MaskBranch(208, num_convs=self.num_convs))
            else:
                self.mask_branch.append(MaskBranch(464, num_convs=self.num_convs))

        self.projection = nn.ModuleList([])
        for i in range(self.n_levels):
            proj = nn.Conv2d(256, self.kernel_dim, kernel_size=1)
            c2_msra_fill(proj)
            self.projection.append(proj)
        
        # instance features.
        self.prior_instance_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                self.prior_instance_branch.append(PriorInstanceBranch(in_channels=208, num_convs=self.num_convs))
            else:
                self.prior_instance_branch.append(PriorInstanceBranch(in_channels=464, num_convs=self.num_convs))

        # instance branch.
        self.instance_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            self.instance_branch.append(InstanceBranch(dim=256, kernel_dim=self.kernel_dim, num_masks=self.num_masks))
        
        

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
            all_iam = []
            all_kernel = []
            all_logits = []
            all_scores = []
            all_masks = []

            for i in range(self.n_levels):
                # if self.coord_conv:
                coord_features = self.compute_coordinates(x)
                x = torch.cat([coord_features, x], dim=1)
                
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
                x = self.up_conv_layers[i](x)
                x = self.up_se_blocks[i](x)
                
                if self.pyramid_pooling:
                    x = torch.cat([x, down_pp_out_tensors[-(i + 1)]], dim=1)
                else:
                    x = torch.cat([x, down_conv_out_tensors[-(i + 1)]], dim=1)
                
                
                if i in [0, 1, 2, 3, 4]:

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
                    
                        
                    logits, kernel, scores, iam = self.instance_branch[i](inst_feats)
                    mask_feats = self.projection[i](mask_feats)

                    B, C, H, W = mask_feats.shape
                    N = kernel.shape[1]
                    level_masks = torch.bmm(
                        kernel, mask_feats.view(B, C, H * W)
                    )
                    level_masks = level_masks.view(B, N, H, W) 

                    all_iam.append(iam)
                    all_kernel.append(kernel)
                    all_logits.append(logits)
                    all_scores.append(scores)
                    all_masks.append(level_masks)

            return all_masks, all_kernel, all_logits, all_scores, all_iam
    
        # cyto
        # x, mask_features, (logits, kernel, scores, iam) = go_up(x)
        masks, kernel, logits, scores, iam = go_up(x)
        
        # Predicting instance masks
        # N = kernel.shape[1]  # num_masks
        # B, C, H, W = mask_features.shape

        # masks = torch.bmm(
        #     kernel,    # (B, N, 128)
        #     mask_features.view(B, C, H * W)   # (B, 128, [HW])
        # ) # -> (B, N, [HW])
        # masks = masks.view(B, N, H, W)  # (B, N, [HW]) -> (B, N, H, W)

        # masks = []
        # for i, k in enumerate(kernel):
        #     B, C, H, W = mask_features.shape

        #     level_masks = torch.bmm(
        #         k, mask_features.view(B, C, H * W)
        #     )
        #     level_masks = level_masks.view(B, N, H, W) 
        #     masks.append(level_masks)


        output = {
            "pred_masks": masks[-1],
            "pred_logits": logits[-1],
            "pred_iam": iam[-1],
            "pred_scores": scores[-1]
        }

        output["aux_outputs"] = [{"pred_masks": m, "pred_logits": l, "pred_iam": i, "pred_scores": c} 
                                 for m, l, i, c in zip(masks[:-1], logits[:-1], iam[:-1], scores[:-1])]
    
        return output



if __name__ == "__main__":
    model = IAUNet(cfg)
    x = torch.rand(2, 3, 512, 512)
    print(model)
    out = model(x)
    print(out["pred_masks"].shape)

    for pred in out["aux_outputs"]:
        print(pred["pred_masks"].shape)
        