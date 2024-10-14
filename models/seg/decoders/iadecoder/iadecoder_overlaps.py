import torch
from torch.nn import functional as F

import sys
sys.path.append("./")

from models.seg.decoders.iadecoder.iadecoder import IADecoder as BaseDecoder
from configs.structure import Decoder
from utils.registry import DECODERS


@DECODERS.register(name='iadecoder_overlaps')
class IADecoder(BaseDecoder):
    def __init__(self, cfg: Decoder, embed_dims: list = [], n_levels: int = 4):
        super().__init__(cfg, embed_dims, n_levels)
        self._init_weights()

    def forward(self, skips, ori_shape):
        results = super()._forward(skips, ori_shape)
        results = self.process_outputs(results, ori_shape)

        return results

    def process_outputs(self, results, ori_shape):
        logits = results["logits"]
        scores = results["objectness_scores"]
        inst_kernel = results["kernels"]["instance_kernel"]
        overlap_kernel = results["kernels"]["overlap_kernel"]
        visible_kernel = results["kernels"]["visible_kernel"]
        bboxes = results["bboxes"]['instance_bboxes']
        mask_feats = results["mask_feats"]
        inst_feats = results["inst_feats"]

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

        output = {
            'pred_logits': logits,
            'pred_scores': scores,
            'pred_iams': results['iams'],
            'pred_instance_masks': inst_masks,
            'pred_overlap_masks': overlap_masks,
            'pred_visible_masks': visible_masks,
            'pred_bboxes': bboxes,
            'pred_instance_feats': {
                "mask_feats": mask_feats,
                "inst_feats": inst_feats
            }
        }

        return output