# copied from 45966022
import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn.weight_init import c2_msra_fill

import sys
sys.path.append("./")

from ...heads.instance_head import Refiner
from models.seg.models.iaunet.iaunet import IAUNet as BaseModel
from configs import cfg
from utils.registry import MODELS, HEADS


@MODELS.register(name="iaunet_ml")
class IAUNet(BaseModel):
    def __init__(self, cfg: cfg):
        super(IAUNet, self).__init__(cfg)

        # instance head.
        self.instance_head = nn.ModuleList([])
        print(self.n_levels)
        for i in range(self.n_levels):
            instance_head = HEADS.build(cfg.model.instance_head)
            self.instance_head.append(instance_head)

        self._init_weights()


    def forward(self, x):
        ori_shape = x.shape

        # go down
        skips = self.encoder(x)

        # middle
        # x = self.bridge(skips[-1])
        x = skips[-1]

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

            if i != 0:
                results = self.instance_head[i](inst_feats, inst_embed)
                inst_embed = results["inst_feats"]
            else:
                results = self.instance_head[i](inst_feats)
                inst_embed = results["inst_feats"]


            # if i == 0:
            #     results = self.instance_head[i](inst_feats, last_stage=False)
            # elif i == self.n_levels - 1:
            #     results = self.instance_head[i](inst_feats, iam, last_stage=True)
            # else:
            #     results = self.instance_head[i](inst_feats, iam, last_stage=False)
            # iam = results["iam"]
                


        logits = results["logits"]
        mask_kernel = results["mask_kernel"]
        scores = results["objectness_scores"]
        bboxes = results["bboxes"]
        iam = results["iam"]

        mask_feats = self.projection(mask_feats)


        # instance masks.
        N = mask_kernel.shape[1]
        B, C, H, W = mask_feats.shape

        inst_masks = torch.bmm(mask_kernel, mask_feats.view(B, C, H * W))
        inst_masks = inst_masks.view(B, N, H, W)
        
        bboxes = bboxes.sigmoid()

        inst_masks = F.interpolate(inst_masks, size=ori_shape[-2:], 
                                   mode="bilinear", align_corners=False)
        iam = F.interpolate(iam, size=ori_shape[-2:], 
                            mode="bilinear", align_corners=False)

        output = {
            'pred_logits': logits,
            'pred_scores': scores,
            'pred_iam': iam,
            'pred_masks': inst_masks,
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