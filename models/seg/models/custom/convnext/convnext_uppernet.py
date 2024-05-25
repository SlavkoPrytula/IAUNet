import torch
from torch import nn
import torch.nn.functional as F

import sys
sys.path.append("./")

from models.seg.heads.instance_head import InstanceBranch, PriorInstanceBranch
from models.seg.heads.mask_head import MaskBranch

from models.seg.encoders.convnext import ConvNeXt
from models.seg.encoders.upernet import UPerNet

from utils.registry import MODELS
from configs import cfg


@MODELS.register(name="convnext")
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
        
        self.encoder = ConvNeXt(in_channels=cfg.model.in_channels, model_name="L")
        weights = torch.load("/gpfs/space/home/prytula/kaggle/models/HuBMAP_HPA_2022/pretrained/convnext/convnext_large_22k_224.pth")
        self.encoder.load_state_dict(weights, strict=False)

        embed_dims = self.encoder.embed_dims
        self.fpn = UPerNet(
            embed_dims, 
            ppm_pool_scale=[1, 2, 3, 6],
            ppm_dim=512,
            fpn_out_dim=512
            )
        
        # mask branch.
        self.mask_branch = nn.ModuleList([])
        self.mask_branch.append(MaskBranch(512+2))
        # for i in range(self.n_levels):
        #     if i == 0:
        #         self.mask_branch.append(MaskBranch(208, num_convs=self.num_convs))
        #     else:
        #         self.mask_branch.append(MaskBranch(208+128, num_convs=self.num_convs))
        
        # instance features.
        self.prior_instance_branch = nn.ModuleList([])
        self.prior_instance_branch.append(PriorInstanceBranch(in_channels=512+2, out_channels=256, num_convs=4))
        # for i in range(self.n_levels):
        #     if i == 0:
        #         self.prior_instance_branch.append(PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=self.num_convs))
        #     else:
        #         self.prior_instance_branch.append(PriorInstanceBranch(in_channels=464, out_channels=256, num_convs=self.num_convs))

        # instance branch.
        self.instance_branch = InstanceBranch(dim=256+2, kernel_dim=self.kernel_dim, num_masks=self.num_masks)
        

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
            
        feats = self.encoder(x)
        x = self.fpn(feats)[0]

        coord_features = self.compute_coordinates(x)
        x = torch.cat([coord_features, x], dim=1)

        mask_features = self.mask_branch[0](x)
        inst_feats = self.prior_instance_branch[0](x)

        coord_features = self.compute_coordinates(inst_feats)
        inst_feats = torch.cat([coord_features, inst_feats], dim=1)

        logits, kernel, scores, iam = self.instance_branch(inst_feats, idx)
        
        # Predicting instance masks
        _, N, D = kernel.shape 
        B, C, H, W = mask_features.shape

        # kernel: (1, N, D) -> (N, D, 1, 1)
        masks = []
        for b in range(len(kernel)):
            m = mask_features[b].unsqueeze(0)
            k = kernel[b]
            k = k.view(N, D, 1, 1)

            inst = F.conv2d(m, k, stride=1)
            masks.append(inst)
        masks = torch.cat(masks, dim=0)

        masks = F.interpolate(masks, size=[512, 512], mode='bilinear', align_corners=False)
        iam = F.interpolate(iam, size=[512, 512], mode='bilinear', align_corners=False)

        iam = {
            "iam": iam,
            }
        
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

