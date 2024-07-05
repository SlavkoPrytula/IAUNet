# copied from 45966022
import torch
from torch import nn

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
        # occluder_kernel = results["occluder_kernel"]
        overlap_kernel = results["overlap_kernel"]
        scores = results["objectness_scores"]
        bboxes = results["bboxes"]
        iam = results["iam"]

        mask_feats = self.projection(mask_feats)

        
        # Predicting instance masks
        N = mask_kernel.shape[1]
        B, C, H, W = mask_feats.shape

        inst_masks = torch.bmm(mask_kernel, mask_feats.view(B, C, H * W))
        inst_masks = inst_masks.view(B, N, H, W)

        # overlap + occluder
        # occluder_masks = torch.bmm(occluder_kernel, mask_feats.view(B, C, H * W))
        # occluder_masks = occluder_masks.view(B, N, H, W)

        overlap_masks = torch.bmm(overlap_kernel, mask_feats.view(B, C, H * W))
        overlap_masks = overlap_masks.view(B, N, H, W)
        
        bboxes = bboxes.sigmoid()

        inst_masks = nn.UpsamplingBilinear2d(scale_factor=4)(inst_masks)
        # occluder_masks = nn.UpsamplingBilinear2d(scale_factor=4)(occluder_masks)
        overlap_masks = nn.UpsamplingBilinear2d(scale_factor=4)(overlap_masks)
        iam = nn.UpsamplingBilinear2d(scale_factor=4)(iam)


        output = {
            'pred_logits': logits,
            'pred_scores': scores,
            'pred_iam': iam,
            'pred_masks': inst_masks,
            # 'pred_occluder_masks': occluder_masks,
            'pred_overlap_masks': overlap_masks,
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