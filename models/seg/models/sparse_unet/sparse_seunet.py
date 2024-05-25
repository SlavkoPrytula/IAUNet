import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn.weight_init import c2_msra_fill

import sys
sys.path.append("./")

from models.seg.heads.instance_head import InstanceBranch, PriorInstanceBranch, DoubleIAMBranch, InstanceBranchNoIAM, IAM
from models.seg.heads.mask_head import MaskBranch


from models.seg.modules.mixup import MixUpScaler
from models.seg.models.base import SparseSEUnet as BaseModel
from models.seg.models.base import DoubleConv, SE_block

from models.seg.blocks import PAM_Module, CoordAtt

from configs import cfg

from utils.registry import MODELS


# TODO: add subnames to registy module (eg. for what is it used)
# TODO: add class path for saving model files
@MODELS.register(name="sparse_unet-deep_supervision-sparse_seunet")
class SparseSEUnet(BaseModel):
    """
    Base SparseUnet + Deep Supervision
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
        self.mask_branch = nn.ModuleList([
            MaskBranch(in_channels=208+2, out_channels=256, num_convs=4)
        ])
        
        # instance features.
        self.prior_instance_branch = nn.ModuleList([
            PriorInstanceBranch(in_channels=208+2, out_channels=256, num_convs=4)
        ])
        
        # -----------------------------------------
        # custom 
        # v0 - deep iam features - deep supervision
        self.instance_iam_convs = nn.ModuleList([
            nn.Sequential(
                PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
                # PAM_Module(256),
                # CoordAtt(256, 256),
                IAM(in_channels=256, out_channels=self.num_masks)
            ), 
            nn.Sequential(
                PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
                IAM(in_channels=256, out_channels=self.num_masks)
            ),
            nn.Sequential(
                PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
                IAM(in_channels=256, out_channels=self.num_masks)
            ),
            nn.Sequential(
                PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
                IAM(in_channels=256, out_channels=self.num_masks)
            ),
            nn.Sequential(
                PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
                IAM(in_channels=256, out_channels=self.num_masks)
            )
        ])

        # -----------------------------------------

        # instance branch.
        self.instance_branch = nn.ModuleList([
            InstanceBranchNoIAM(dim=256, kernel_dim=self.kernel_dim),
            InstanceBranchNoIAM(dim=256, kernel_dim=self.kernel_dim),
            InstanceBranchNoIAM(dim=256, kernel_dim=self.kernel_dim),
            InstanceBranchNoIAM(dim=256, kernel_dim=self.kernel_dim),
            InstanceBranchNoIAM(dim=256, kernel_dim=self.kernel_dim),
        ])

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
            iam = []
            kernel = []
            logits = []

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
                    inst_iam = self.instance_iam_convs[i](x)
                    iam.append(inst_iam)
                
                
            coord_features = self.compute_coordinates(x)
            x = torch.cat([coord_features, x], dim=1)

            # mask branch.
            mb = self.mask_branch[0](x)
            mb = self.projection(mb)
            
            inst_feats = self.prior_instance_branch[0](x)
            
            for j, _iam in enumerate(iam):
                # iam[j] = F.interpolate(iam[j], size=mb.shape[-2:], mode="bilinear", align_corners=False)
                # inst_logits, inst_kernel, scores, inst_iam = self.instance_branch[j](inst_feats, iam[j])

                _iam = F.interpolate(_iam, size=mb.shape[-2:], mode="bilinear", align_corners=False)
                inst_logits, inst_kernel, scores, inst_iam, _ = self.instance_branch[j](inst_feats, _iam)

                kernel.append(inst_kernel)
                logits.append(inst_logits)

            return x, mb, (logits, kernel, scores, iam)
    
        # cyto
        x, mask_features, (logits, kernel, scores, iam) = go_up(x)
        
        # Predicting instance masks
        N = self.num_masks
        B, C, H, W = mask_features.shape

        masks = []
        for i, k in enumerate(kernel):
            level_masks = torch.bmm(
                k, mask_features.view(B, C, H * W)
            ) 
            level_masks = level_masks.view(B, N, H, W) 
            masks.append(level_masks)

        output = {
            "pred_masks": masks[-1],
            "pred_logits": logits[-1],
            "pred_iam": iam[-1],
        }

        output["aux_outputs"] = [{"pred_masks": m, "pred_logits": l, "pred_iam": i} 
                                 for m, l, i in zip(masks[:-1], logits[:-1], iam[:-1])]
    
        return output


if __name__ == "__main__":
    model = SparseSEUnet(cfg)
    x = torch.rand(2, 2, 512, 512)
    out = model(x)
    print(out["pred_masks"].shape)




