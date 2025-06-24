import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn.weight_init import c2_msra_fill

import sys
sys.path.append("./")

from models.seg.heads.instance_head import InstanceBranch, PriorInstanceBranch, DoubleIAMBranch, InstanceBranchNoIAM, IAM
from models.seg.heads.mask_head import MaskBranch


from models.seg.nn.modules.mixup import MixUpScaler
from models.seg.models.base import SparseSEUnet as BaseModel
from models.seg.models.base import DoubleConv, SE_block

from configs import cfg

from utils.registry import MODELS


@MODELS.register(name="occluders-sparse_seunet_occluder")
class SparseSEUnet(BaseModel):
    """
    Base SparseUnet + Simple Instance-Occluder interaction
    """
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

        self.up_conv_layers = nn.ModuleList([])
        for _ in range(self.n_levels):
            if len(self.up_conv_layers) == 0:
                upconv = DoubleConv(self.n_filters + 2, self.n_filters)
            else:
                upconv = DoubleConv(
                    (self.n_filters // 4) * 5 + 2 * self.n_filters + 2, self.n_filters
                )
            self.up_conv_layers.append(upconv)

        
        # mask branch.
        self.mask_branch = nn.ModuleList([])
        self.mask_branch.append(MaskBranch(in_channels=208+2, out_channels=256, num_convs=4))
        
        # instance features.
        self.prior_instance_branch = nn.ModuleList([
            PriorInstanceBranch(in_channels=208+2, out_channels=256, num_convs=1),
            PriorInstanceBranch(in_channels=256, out_channels=256, num_convs=3)
        ])
        self.prior_occluder_branch = nn.ModuleList([
            PriorInstanceBranch(in_channels=208+2, out_channels=256, num_convs=1),
            PriorInstanceBranch(in_channels=256, out_channels=256, num_convs=3)
        ])
        
        # -----------------------------------------
        # custom 
        # v0 - deep iam features - deep supervision
        self.instance_iam_convs = nn.Module(
            nn.Sequential(
                PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
                IAM(in_channels=256, out_channels=self.num_masks)
            ), 
            nn.Sequential(
                PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
                IAM(in_channels=256, out_channels=self.num_masks)
            )
        )


        # self.instance_iam_conv = nn.Sequential(
        #     PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
        #     IAM(in_channels=256, out_channels=self.num_masks)
        # )
        # self.occluder_iam_conv = nn.Sequential(
        #     PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
        #     IAM(in_channels=256, out_channels=self.num_masks)
        # )
        # -----------------------------------------

        # instance branch.
        self.instance_branch = InstanceBranchNoIAM(dim=256, kernel_dim=self.kernel_dim)
        # self.occluder_branch = InstanceBranchNoIAM(dim=256, kernel_dim=self.kernel_dim)

        # self.instance_branch = DoubleIAMBranch(dim=256, kernel_dim=self.kernel_dim, num_masks=self.num_masks)
        # self.occluder_branch = InstanceBranch(dim=256, kernel_dim=self.kernel_dim, num_masks=self.num_masks)
        
        # self.instance_branch = InstanceBranch(dim=256, kernel_dim=self.kernel_dim, num_masks=self.num_masks)
        # self.occluder_branch = InstanceBranch(dim=256, kernel_dim=self.kernel_dim, num_masks=self.num_masks)

        self.projection = nn.Conv2d(256, self.kernel_dim, kernel_size=1)
        c2_msra_fill(self.projection)
        

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

                
                if i == 2:
                    inst_iam = self.instance_iam_conv(x)
                    occl_iam = self.occluder_iam_conv(x)
                
                
            coord_features = self.compute_coordinates(x)
            x = torch.cat([coord_features, x], dim=1)

            # mask branch.
            mb = self.mask_branch[0](x)
            mb = self.projection(mb)


            # occluder branch.
            # occl_feats = self.prior_occluder_branch[0](x)           # (B, 208+2, H, W) -> (B, 208+2, H, W)
            # occl_feats = self.prior_occluder_branch[1](occl_feats)  # (B, 208+2, H, W) -> (B, 208+2, H, W)

            # instance branch.
            inst_feats = self.prior_instance_branch[0](x)  # (B, 208+2, H, W) -> (B, 208+2, H, W)
            # inst_feats = inst_feats + occl_feats                             # (B, 208+2, H, W)
            inst_feats = self.prior_instance_branch[1](inst_feats)  # (B, 208+2, H, W) -> (B, 208+2, H, W)




            # inst_logits, inst_kernel, inst_scores, inst_iam = self.instance_branch(inst_feats, idx)
            # occl_logits, occl_kernel, occl_scores, occl_iam = self.occluder_branch(occl_feats, idx)

            # inst_logits, inst_kernel, inst_scores, inst_iam, occl_logits, occl_kernel, occl_scores, occl_iam = self.instance_branch(inst_feats, occl_feats, idx)

            inst_iam = F.interpolate(inst_iam, size=mb.shape[-2:], mode="bilinear", align_corners=False)
            occl_iam = F.interpolate(occl_iam, size=mb.shape[-2:], mode="bilinear", align_corners=False)
            inst_logits, inst_kernel, inst_scores, inst_iam = self.instance_branch(inst_feats, inst_iam)
            # occl_logits, occl_kernel, occl_scores, occl_iam = self.instance_branch(occl_feats, occl_iam)


            mb = {
                "mb": mb,
            }
            
            logits = {
                # "occluder_logits": occl_logits,
                "instance_logits": inst_logits,
            }
            scores = {
                # "occluder_scores": occl_scores,
                "instance_scores": inst_scores,
            }

            kernel = {
                # "occluder_kernel": occl_kernel,
                "instance_kernel": inst_kernel,
            }

            iam = {
                # "occluder_iam": occl_iam,
                "instance_iam": inst_iam,
            }

            return x, mb, (logits, kernel, scores, iam)
    
        # cyto
        x, mask_features, (logits, kernel, scores, iam) = go_up(x)
        
        mask_features = mask_features["mb"]
        
        inst_kernel = kernel["instance_kernel"]
        # occl_kernel = kernel["occluder_kernel"]

        # Predicting instance masks
        N = inst_kernel.shape[1]  # num_masks
        B, C, H, W = mask_features.shape

        masks = torch.bmm(
            inst_kernel, 
            mask_features.view(B, C, H * W)
        ) 
        masks = masks.view(B, N, H, W) 

        # occluders = torch.bmm(
        #     occl_kernel,    
        #     mask_features.view(B, C, H * W)   
        # ) 
        # occluders = occluders.view(B, N, H, W) 


        inst_logits = logits["instance_logits"]
        occl_logits = logits["occluder_logits"]

        inst_scores = scores["instance_scores"]
        occl_scores = scores["occluder_scores"]


        output = {
            'pred_logits': inst_logits,
            'pred_scores': inst_scores,
            'pred_iam': iam,
            'pred_masks': masks,  # instnace masks
            'pred_kernel': kernel,
            'pred_occluders': occluders, # occluders masks
            'pred_logits_occluders': occl_logits,
            'pred_scores_occluders': occl_scores,
        }
    
        return output


if __name__ == "__main__":
    model = SparseSEUnet(cfg)
    x = torch.rand(2, 2, 512, 512)
    out = model(x)
    print(out["pred_masks"].shape)
