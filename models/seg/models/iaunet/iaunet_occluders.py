# copied from 45966022
import torch
from torch import nn
from torch.nn import functional as F

import sys
sys.path.append("./")

from .iaunet import IAUNet as BaseModel
from configs import cfg
from utils.registry import MODELS


@MODELS.register(name="iaunet_occluders")
class IAUNet(BaseModel):
    def __init__(self, cfg: cfg):
        super(IAUNet, self).__init__(cfg)
        self._init_weights()

    def forward(self, x):
        ori_shape = x.shape

        skips = self.encoder(x)
        x = skips[-1]

        for i in range(self.n_levels):
            if i != 0:
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
            coord_features = self.compute_coordinates(x)
            x = torch.cat([coord_features, x], dim=1)

            x = torch.cat([x, skips[-(i + 1)]], dim=1)
            x = self.up_conv_layers[i](x)

            
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
        scores = results["objectness_scores"]
        inst_kernel = results["kernels"]["instance_kernel"]
        overlap_kernel = results["kernels"]["overlap_kernel"]
        visible_kernel = results["kernels"]["visible_kernel"]
        bboxes = results["bboxes"]['instance_bboxes']

        mask_feats = self.projection(mask_feats)


        # instance masks.
        N = inst_kernel.shape[1]
        B, C, H, W = mask_feats.shape

        inst_masks = torch.bmm(inst_kernel, mask_feats.view(B, C, H * W))
        inst_masks = inst_masks.view(B, N, H, W)

        overlap_masks = torch.bmm(overlap_kernel, mask_feats.view(B, C, H * W))
        overlap_masks = overlap_masks.view(B, N, H, W)

        visible_masks = torch.bmm(visible_kernel, mask_feats.view(B, C, H * W))
        visible_masks = visible_masks.view(B, N, H, W)

        bboxes = bboxes.sigmoid()

        inst_masks = F.interpolate(inst_masks, size=ori_shape[-2:], 
                                   mode="bilinear", align_corners=False)
        overlap_masks = F.interpolate(overlap_masks, size=ori_shape[-2:], 
                                      mode="bilinear", align_corners=False)
        visible_masks = F.interpolate(visible_masks, size=ori_shape[-2:], 
                                      mode="bilinear", align_corners=False)
        # iam = F.interpolate(iam, size=ori_shape[-2:], 
        #                     mode="bilinear", align_corners=False)

        output = {
            'pred_logits': logits,
            'pred_scores': scores,
            'pred_iams': results['iams'],
            'pred_instance_masks': inst_masks,
            'pred_overlap_masks': overlap_masks,
            'pred_visible_masks': visible_masks,
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
    print(out["pred_occluder_masks"].shape)
    print(out["pred_overlap_masks"].shape)
    time_e = time.time()
    print(f'loaded in {time_e - time_s}(s)')