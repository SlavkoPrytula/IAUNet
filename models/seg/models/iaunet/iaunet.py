# copied from 45966022
import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn.weight_init import c2_msra_fill

import sys
sys.path.append("./")

# from models.seg.heads.instance_head import InstanceHead, InstanceBranch
# from models.seg.heads.mask_head import MaskBranch

# from models.seg.models.base import DoubleConv, SE_block
# from models.seg.blocks.tests import (PyramidPooling, PyramidPooling_v3, PyramidPooling_v5, 
#                              DoubleConv_v2, DoubleConv_v3, DoubleConvModule)


from ...heads.instance_head import InstanceHead, InstanceBranch
from ...heads.mask_head import MaskBranch

from models.seg.models.base import BaseModel
from models.seg.models.base import DoubleConv, SE_block
from ...nn.blocks import (PyramidPooling, PyramidPooling_v3, PyramidPooling_v5, 
                             DoubleConv_v2, DoubleConv_v3, DoubleConvModule)

from configs import cfg
from utils.registry import MODELS, HEADS



@MODELS.register(name="iaunet")
class IAUNet(BaseModel):
    def __init__(self, cfg: cfg):
        super(IAUNet, self).__init__(cfg)

        self.encoder = MODELS.build(cfg.model.backbone)
        embed_dims = self.encoder.embed_dims
        self.embed_dims = embed_dims

        self.bridge = nn.Sequential(
            DoubleConv_v2(embed_dims[-1], embed_dims[-1]),
            SE_block(num_features=embed_dims[-1]),
        )
        
        embed_dims = embed_dims[::-1]
        self.up_conv_layers = nn.ModuleList([])
        for i in range(self.n_levels):
            in_channels = embed_dims[i] * 2 + 2

            if i != self.n_levels-1:
                out_channels = embed_dims[i+1]
            else:
                out_channels = embed_dims[i]
                
            upconv = nn.Sequential(
                DoubleConv_v2(in_channels, out_channels),
                SE_block(num_features=out_channels)
            )
            self.up_conv_layers.append(upconv)

        embed_dims = embed_dims[1:] + [embed_dims[-1]]

        # mask branch.
        mask_dim = cfg.model.mask_dim
        self.mask_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                mask_branch = MaskBranch(
                    embed_dims[i], 
                    out_channels=mask_dim, 
                    num_convs=self.num_convs
                    )
            else:
                mask_branch = MaskBranch(
                    embed_dims[i] + mask_dim, 
                    out_channels=mask_dim, 
                    num_convs=self.num_convs
                    )
            self.mask_branch.append(mask_branch)
        self.projection = nn.Conv2d(mask_dim, self.kernel_dim, kernel_size=1)
        
        # instance branch.
        self.instance_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                instance_branch = InstanceBranch(
                    in_channels=embed_dims[i] + 2, 
                    out_channels=mask_dim, 
                    num_convs=self.num_convs
                    )
            else:
                instance_branch = InstanceBranch(
                    in_channels=embed_dims[i] + mask_dim + 2, 
                    out_channels=mask_dim, 
                    num_convs=self.num_convs
                    )
            self.instance_branch.append(instance_branch)

        # instance head.
        self.instance_head = HEADS.build(cfg.model.instance_head)

        self._init_weights()


    def _init_weights(self):
        for modules in [self.up_conv_layers]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

        c2_msra_fill(self.projection)
        

    def forward(self, x):
        # go down
        skips = self.encoder(x)

        # middle
        x = self.bridge(skips[-1])

        # go up
        for i in range(self.n_levels):
            if i != 0:
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
            coord_features = self.compute_coordinates(x)
            x = torch.cat([coord_features, x], dim=1)

            x = torch.cat([x, skips[-(i + 1)]], dim=1)
            x = self.up_conv_layers[i](x)

            
            # multi-level
            # if self.multi_level:
            if i != 0:
                mask_feats = nn.UpsamplingBilinear2d(scale_factor=2)(mask_feats)
                mask_feats = torch.cat([x, mask_feats], dim=1)
                mask_feats = self.mask_branch[i](mask_feats)
            else:
                mask_feats = self.mask_branch[i](x)

            if i != 0:
                inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
                inst_feats = torch.cat([x, inst_feats], dim=1)
                coord_features = self.compute_coordinates(inst_feats)

                inst_feats = torch.cat([coord_features, inst_feats], dim=1)
                inst_feats = self.instance_branch[i](inst_feats)
            else:
                coord_features = self.compute_coordinates(x)
                inst_feats = torch.cat([coord_features, x], dim=1)
                inst_feats = self.instance_branch[i](inst_feats)
                    

        # out layer.
        results = self.instance_head(inst_feats)

        logits = results["logits"]
        mask_kernel = results["mask_kernel"]
        # border_kernel = results["border_kernel"]
        scores = results["objectness_scores"]
        bboxes = results["bboxes"]
        iam = results["iam"]

        mask_feats = self.projection(mask_feats)

        
        # Predicting instance masks
        N = mask_kernel.shape[1]
        B, C, H, W = mask_feats.shape

        inst_masks = torch.bmm(mask_kernel, mask_feats.view(B, C, H * W))
        inst_masks = inst_masks.view(B, N, H, W)

        # borders_masks = torch.bmm(border_kernel, mask_feats.view(B, C, H * W))
        # borders_masks = borders_masks.view(B, N, H, W)
        
        bboxes = bboxes.sigmoid()

        inst_masks = nn.UpsamplingBilinear2d(scale_factor=2)(inst_masks)
        # borders_masks = nn.UpsamplingBilinear2d(scale_factor=2)(borders_masks)
        iam = nn.UpsamplingBilinear2d(scale_factor=2)(iam)


        output = {
            'pred_logits': logits,
            'pred_scores': scores,
            'pred_iam': iam,
            'pred_masks': inst_masks,
            # 'pred_borders_masks': borders_masks,
            'pred_bboxes': bboxes,
        }
    
        return output
    

if __name__ == "__main__":
    import time 

    model = IAUNet(cfg)
    x = torch.rand(1, 3, 512, 512)
    
    time_s = time.time()
    out = model(x)
    print(out["pred_masks"].shape)
    time_e = time.time()
    print(f'loaded in {time_e - time_s}(s)')