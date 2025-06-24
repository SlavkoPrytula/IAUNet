import torch
from torch import nn
import torch.nn.functional as F

import sys
sys.path.append("./")

from models.seg.heads.instance_head import InstanceBranch, PriorInstanceBranch, GroupInstanceBranch, DoubleIAMBranch
from models.seg.heads.mask_head import MaskBranch

from models.seg.encoders.hornet import HorNet
from models.seg.encoders.upernet import UPerNet

from utils.registry import MODELS
from configs import cfg


@MODELS.register(name="hornet_occluder")
class SparseSEUnet(nn.Module):
    def __init__(
        self,
        cfg: cfg
    ):
        super(SparseSEUnet, self).__init__()  
        
        self.coord_conv = cfg.model.coord_conv
        self.multi_level = cfg.model.multi_level
        self.kernel_dim = cfg.model.kernel_dim
        self.num_masks = cfg.model.num_masks
        self.num_convs = cfg.model.num_convs
        
        self.encoder = HorNet(in_chans=cfg.model.in_channels, num_classes=1)
        embed_dims = self.encoder.embed_dims
        self.fpn = UPerNet(
            embed_dims, 
            ppm_pool_scale=[1, 2, 3, 6],
            ppm_dim=512,
            fpn_out_dim=512
            )
        
        # mask branch.
        self.mask_branch_instance = MaskBranch(512)
        self.mask_branch_occluder = MaskBranch(512)
        # self.mask_branch = nn.ModuleList([])
        # self.mask_branch.append(MaskBranch(512))
        # for i in range(self.n_levels):
        #     if i == 0:
        #         self.mask_branch.append(MaskBranch(208, num_convs=self.num_convs))
        #     else:
        #         self.mask_branch.append(MaskBranch(208+128, num_convs=self.num_convs))
        
        # instance features.
        self.prior_instance_branch = PriorInstanceBranch(in_channels=512, out_channels=256, num_convs=4)
        self.prior_occluder_branch = PriorInstanceBranch(in_channels=512, out_channels=256, num_convs=4)
        # self.prior_instance_branch = nn.ModuleList([])
        # self.prior_instance_branch.append(PriorInstanceBranch(in_channels=512, out_channels=256, num_convs=4))
        # for i in range(self.n_levels):
        #     if i == 0:
        #         self.prior_instance_branch.append(PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=self.num_convs))
        #     else:
        #         self.prior_instance_branch.append(PriorInstanceBranch(in_channels=464, out_channels=256, num_convs=self.num_convs))

        # instance branch.
        # self.instance_branch = GroupInstanceBranch(dim=256, kernel_dim=self.kernel_dim, num_masks=self.num_masks)
        # self.occluder_branch = GroupInstanceBranch(dim=256, kernel_dim=self.kernel_dim, num_masks=self.num_masks)

        self.instance_branch = DoubleIAMBranch(
            dim=256, 
            kernel_dim=self.kernel_dim, 
            num_masks=self.num_masks
            )
        
        

    # TESTING: add instance and mask branches only to the final layer of the decoder
    def forward(self, x, idx=None):
            
        feats = self.encoder(x)
        x = self.fpn(feats)[0]

        occl_feats = x.clone()
        inst_feats = x.clone()

        mask_features_occl = self.mask_branch_occluder(x)
        mask_features_inst = self.mask_branch_instance(x)

        occl_feats = self.prior_occluder_branch(x)
        inst_feats = self.prior_instance_branch(x)

        # occl_logits, occl_kernel, occl_scores, occl_iam = self.occluder_branch(occl_feats)
        # inst_logits, inst_kernel, inst_scores, inst_iam = self.instance_branch(inst_feats)

        inst_logits, inst_kernel, inst_scores, inst_iam, occl_logits, occl_kernel, occl_scores, occl_iam = self.instance_branch(inst_feats, occl_feats)

        # Predicting instance masks
        N = inst_kernel.shape[1]  # num_masks
        B, C, H, W = mask_features_inst.shape

        masks = torch.bmm(
            inst_kernel,    # (B, N, 128)
            mask_features_inst.view(B, C, H * W)   # (B, 128, [HW])
        ) # -> (B, N, [HW])
        masks = masks.view(B, N, H, W)  # (B, N, [HW]) -> (B, N, H, W)

        occluders = torch.bmm(
            occl_kernel,    # (B, N, 128)
            mask_features_occl.view(B, C, H * W)   # (B, 128, [HW])
        ) # -> (B, N, [HW])
        occluders = occluders.view(B, N, H, W)  # (B, N, [HW]) -> (B, N, H, W)


        masks = F.interpolate(masks, size=[512, 512], mode='bilinear', align_corners=False)
        occluders = F.interpolate(occluders, size=[512, 512], mode='bilinear', align_corners=False)
        inst_iam = F.interpolate(inst_iam, size=[512, 512], mode='bilinear', align_corners=False)
        occl_iam = F.interpolate(occl_iam, size=[512, 512], mode='bilinear', align_corners=False)


        iam = {
            "occluder_iam": occl_iam,
            "instance_iam": inst_iam,
        }

        # TODO: rename parameters
        # TODO: pass in this format {"pred_masks": {"inst": ..., "occluder": ...}, "logits": {...}}
        output = {
            'pred_logits': inst_logits,
            'pred_scores': inst_scores,
            'pred_iam': iam,
            'pred_masks': masks,  # instnace masks
            'pred_occluders': occluders, # occluders masks
            'pred_logits_occluders': occl_logits,
            'pred_scores_occluders': occl_scores,
        }

        return output



if __name__ == "__main__":
    model = SparseSEUnet(cfg)
    x = torch.rand(1, 1, 512, 512)
    out = model(x)
    print(out["pred_masks"].shape)

