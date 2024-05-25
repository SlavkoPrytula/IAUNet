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
@MODELS.register(name="sparse_unet_occluder-deep_supervision-sparse_seunet")
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
            MaskBranch(in_channels=208, out_channels=256, num_convs=4)
        ])
        
        # instance features.
        self.prior_instance_branch = nn.ModuleList([
            nn.ModuleList([
                PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
            ]),
            nn.ModuleList([
                PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
            ]),
            nn.ModuleList([
                PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
            ]),
            nn.ModuleList([
                PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
            ]),
            nn.ModuleList([
                PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
            ]),
        ])

        self.prior_occluder_branch = nn.ModuleList([
            nn.ModuleList([
                PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
            ]),
            nn.ModuleList([
                PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
            ]),
            nn.ModuleList([
                PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
            ]),
            nn.ModuleList([
                PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
            ]),
            nn.ModuleList([
                PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
            ]),
        ])
        
        # -----------------------------------------
        # custom 
        # v0 - deep iam features - deep supervision
        # self.instance_iam_convs = nn.ModuleList([
        #     nn.Sequential(
        #         PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
        #         IAM(in_channels=256, out_channels=self.num_masks)
        #     ), 
        #     nn.Sequential(
        #         PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
        #         IAM(in_channels=256, out_channels=self.num_masks)
        #     ),
        #     nn.Sequential(
        #         PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
        #         IAM(in_channels=256, out_channels=self.num_masks)
        #     ),
        #     nn.Sequential(
        #         PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
        #         IAM(in_channels=256, out_channels=self.num_masks)
        #     ),
        #     nn.Sequential(
        #         PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
        #         IAM(in_channels=256, out_channels=self.num_masks)
        #     )
        # ])


        # self.occluder_iam_convs = nn.ModuleList([
        #     nn.Sequential(
        #         PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
        #         IAM(in_channels=256, out_channels=self.num_masks)
        #     ), 
        #     nn.Sequential(
        #         PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
        #         IAM(in_channels=256, out_channels=self.num_masks)
        #     ),
        #     nn.Sequential(
        #         PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
        #         IAM(in_channels=256, out_channels=self.num_masks)
        #     ),
        #     nn.Sequential(
        #         PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
        #         IAM(in_channels=256, out_channels=self.num_masks)
        #     ),
        #     nn.Sequential(
        #         PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4),
        #         IAM(in_channels=256, out_channels=self.num_masks)
        #     )
        # ])

        # -----------------------------------------

        # instance branch.
        self.instance_branch = nn.ModuleList([
            InstanceBranch(dim=256, kernel_dim=self.kernel_dim, num_masks=self.num_masks, out_dim=1024),
            InstanceBranch(dim=256, kernel_dim=self.kernel_dim, num_masks=self.num_masks, out_dim=1024),
            InstanceBranch(dim=256, kernel_dim=self.kernel_dim, num_masks=self.num_masks, out_dim=1024),
            InstanceBranch(dim=256, kernel_dim=self.kernel_dim, num_masks=self.num_masks, out_dim=1024),
            InstanceBranch(dim=1280, kernel_dim=self.kernel_dim, num_masks=self.num_masks, out_dim=1024),
        ])

        self.occluder_branch = nn.ModuleList([
            InstanceBranch(dim=256, kernel_dim=self.kernel_dim, num_masks=self.num_masks, out_dim=1024),
            InstanceBranch(dim=256, kernel_dim=self.kernel_dim, num_masks=self.num_masks, out_dim=1024),
            InstanceBranch(dim=256, kernel_dim=self.kernel_dim, num_masks=self.num_masks, out_dim=1024),
            InstanceBranch(dim=256, kernel_dim=self.kernel_dim, num_masks=self.num_masks, out_dim=1024),
            InstanceBranch(dim=1280, kernel_dim=self.kernel_dim, num_masks=self.num_masks, out_dim=1024),
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
            instance_iam = []
            occluder_iam = []

            instance_kernel = []
            occluder_kernel = []

            instance_logits = []
            occluder_logits = []

            instance_scores = []
            occluder_scores = []

            instance_coords = []
            occluder_coords = []

            r = [1, 2, 4, 8, 16]

            instance_feats = []
            occluder_feats = []

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
                    occl_feats = self.prior_occluder_branch[i][0](x)
                    inst_feats = self.prior_instance_branch[i][0](x)

                    occluder_feats.append(F.interpolate(occl_feats, size=[512, 512], mode="bilinear", align_corners=False))
                    instance_feats.append(F.interpolate(inst_feats, size=[512, 512], mode="bilinear", align_corners=False))

                if i in [4]:
                    # occl_feats = self.prior_occluder_branch[i][0](x)
                    # inst_feats = self.prior_instance_branch[i][0](x)
                    occl_feats = torch.cat(occluder_feats, dim=1)
                    inst_feats = torch.cat(instance_feats, dim=1)

                    occl_logits, occl_kernel, occl_scores, occl_iam, occl_coord = self.occluder_branch[i](occl_feats, reduction=r[i])
                    inst_logits, inst_kernel, inst_scores, inst_iam, inst_coord = self.instance_branch[i](inst_feats, reduction=r[i])

                    occluder_iam.append(occl_iam)
                    instance_iam.append(inst_iam)

                    occluder_logits.append(occl_logits)
                    instance_logits.append(inst_logits)

                    occluder_scores.append(occl_scores)
                    instance_scores.append(inst_scores)

                    occluder_kernel.append(occl_kernel)
                    instance_kernel.append(inst_kernel)

                    occluder_coords.append(occl_coord)
                    instance_coords.append(inst_coord)


            # mask branch.
            mb = self.mask_branch[0](x)
            mb = self.projection(mb)
            

            return x, mb, (occluder_logits, instance_logits, 
                           occluder_kernel, instance_kernel, 
                           occluder_scores, instance_scores, 
                           occluder_iam, instance_iam, 
                           occluder_coords, instance_coords)
    
        # cyto
        x, mask_features, (occluder_logits, instance_logits, 
                            occluder_kernel, instance_kernel, 
                            occluder_scores, instance_scores, 
                            occluder_iam, instance_iam, 
                            occluder_coords, instance_coords) = go_up(x)
        
        # Predicting instance masks
        N = self.num_masks
        B, C, H, W = mask_features.shape

        instance_masks = []
        for i, k in enumerate(instance_kernel):
            level_masks = torch.bmm(
                k, mask_features.view(B, C, H * W)
            ) 
            level_masks = level_masks.view(B, N, H, W) 
            instance_masks.append(level_masks)

        occluder_masks = []
        for i, k in enumerate(occluder_kernel):
            level_masks = torch.bmm(
                k, mask_features.view(B, C, H * W)
            ) 
            level_masks = level_masks.view(B, N, H, W) 
            occluder_masks.append(level_masks)


        output = {
            "pred_masks": instance_masks[-1],
            "pred_logits": instance_logits[-1],
            "pred_iam": instance_iam[-1],
            "pred_coords": instance_coords[-1],

            "pred_occluders_masks": occluder_masks[-1],
            "pred_occluders_logits": occluder_logits[-1],
            "pred_occluders_iam": occluder_iam[-1],
            "pred_occluders_coords": occluder_coords[-1],
        }

        # output["aux_outputs"] = [{
        #     "pred_masks": inst_m, 
        #     "pred_logits": inst_l, 
        #     "pred_iam": inst_i,
        #     "pred_coords": inst_c,

        #     "pred_occluders_masks": occl_m, 
        #     "pred_occluders_logits": occl_l, 
        #     "pred_occluders_iam": occl_i,
        #     "pred_occluders_coords": occl_c
        #     } 
        #     for inst_m, inst_l, inst_i, inst_c, occl_m, occl_l, occl_i, occl_c 
        #     in zip(instance_masks[:-1], instance_logits[:-1], instance_iam[:-1], instance_coords[:-1],
        #            occluder_masks[:-1], occluder_logits[:-1], occluder_iam[:-1], occluder_coords[:-1])]
    
        return output


if __name__ == "__main__":
    model = SparseSEUnet(cfg)
    x = torch.rand(2, 2, 512, 512)
    out = model(x)
    print(out["pred_masks"].shape)




