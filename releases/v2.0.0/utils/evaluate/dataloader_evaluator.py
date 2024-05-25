import numpy as np
import torch
from torch import nn

# from utils.coco.coco import COCO
from utils.coco.mask2coco import masks2coco
# from pycocotools.cocoeval import COCOeval

from .coco_evaluator import Evaluator
from utils.utils import nested_tensor_from_tensor_list

from configs import cfg
import matplotlib.pyplot as plt

from utils.utils import flatten_mask
from utils.opt.mask_nms import mask_nms

from utils.registry import EVALUATORS


# TODO: merge base and nms evaluators
@EVALUATORS.register(name="DataloaderEvaluator")
class DataloaderEvaluator(Evaluator):
    # coco_eval
    def __init__(self, cfg: cfg):
        super(DataloaderEvaluator, self).__init__(cfg)

    def forward(self, model, dataloader):
        gt_masks = []
        pred_masks = []
        pred_scores = []

        for step, batch in enumerate(dataloader):
            # prepare targets
            images = []
            targets = []

            # for target in batch:
            target = batch[0]
            target = {k: v.to(cfg.device) for k, v in target.items()}
            images.append(target["image"])
            targets.append(target)

            image = nested_tensor_from_tensor_list(images)   # (B, C, H, W)

            # predict.
            output = self.inference_single(model, image.tensors)

            pred = output
            scores = pred['pred_logits'].sigmoid()
            scores = scores[0, :, 0]

            masks_pred = pred['pred_masks'].sigmoid()
            masks_pred = masks_pred[0, ...]
            
            # maskness scores.
            maskness_scores = []
            for p in masks_pred:
                maskness_score = torch.mean(p[p.gt(self.mask_threshold)])
                maskness_score = torch.nan_to_num(maskness_score, nan=0.0)
                maskness_score = maskness_score.cpu()
                maskness_scores.append(maskness_score)

            maskness_scores = torch.tensor(maskness_scores).to(cfg.device)
            scores = maskness_scores

            # masks_pred = masks_pred[scores > 0.4]
            # scores = scores[scores > 0.4]

            scores = scores.detach().cpu().numpy()
            masks_pred = masks_pred.detach().cpu().numpy()
            masks_pred = (masks_pred > self.mask_threshold).astype(np.uint8)

            masks = target['masks']
            masks = masks.detach().cpu().numpy()

            # store data.
            gt_masks.append(masks)
            pred_masks.append(masks_pred)
            pred_scores.append(scores)

        # masks2coco
        self.gt_coco = masks2coco(gt_masks)
        self.pred_coco = masks2coco(pred_masks, scores=pred_scores)



# @EVALUATORS.register(name="DataloaderEvaluatorNMS")
# class DataloaderEvaluatorNMS(Evaluator):
#     # coco_eval - nms
#     def __init__(self, cfg: cfg):
#         super(DataloaderEvaluatorNMS, self).__init__(cfg)

#     def forward(self, model, dataloader):
#         gt_masks = []
#         pred_masks = []
#         pred_scores = []

#         for step, batch in enumerate(dataloader):
#             # prepare targets
#             images = []
#             targets = []

#             # for target in batch:
#             target = batch[0]
#             target = {k: v.to(cfg.device) for k, v in target.items()}
#             images.append(target["image"])
#             targets.append(target)

#             image = nested_tensor_from_tensor_list(images)   # (B, C, H, W)

#             # predict.
#             output = self.inference_single(model, image.tensors)

#             pred = output
#             scores = pred['pred_logits'].sigmoid()
#             scores = scores[0, :, 0]

#             masks_pred = pred['pred_masks'].sigmoid()
#             masks_pred = masks_pred[0, ...]

#             N, H, W = masks_pred.shape
            
#             # maskness scores.
#             # maskness_scores = []
#             # for p in masks_pred:
#             #     maskness_score = torch.mean(p[p.gt(self.mask_threshold)])
#             #     maskness_score = torch.nan_to_num(maskness_score, nan=0.0)
#             #     maskness_score = maskness_score.cpu()
#             #     maskness_scores.append(maskness_score)

#             # maskness_scores = torch.tensor(maskness_scores).to(cfg.device)
#             # scores = maskness_scores

#             # masks_pred = masks_pred[scores > 0.2]
#             # scores = scores[scores > 0.2]

#             # scores = scores.detach().cpu().numpy()
#             # masks_pred = masks_pred.detach().cpu().numpy()
#             # masks_pred = (masks_pred > self.mask_threshold).astype(np.uint8)

#             seg_masks = masks_pred > self.mask_threshold
#             sum_masks = seg_masks.sum((1, 2)).float()

#             # maskness scores.
#             maskness_scores = (masks_pred * seg_masks.float()).sum((1, 2)) / sum_masks

#             # sort predictions
#             sort_inds = torch.argsort(maskness_scores, descending=True)
#             seg_masks = seg_masks[sort_inds, :, :]
#             masks_pred = masks_pred[sort_inds, :, :]
#             sum_masks = sum_masks[sort_inds]
#             maskness_scores = maskness_scores[sort_inds]
#             scores = scores[sort_inds]
#             labels = torch.ones(N)

#             # nms
#             keep = mask_nms(labels, seg_masks, sum_masks, maskness_scores, nms_thr=0.5)
#             masks_pred = masks_pred[keep, :, :]
#             maskness_scores = maskness_scores[keep]
#             scores = scores[keep]

#             maskness_scores = maskness_scores.detach().cpu().numpy()

#             masks_pred = masks_pred.detach().cpu().numpy()
#             masks_pred = (masks_pred > self.mask_threshold).astype(np.uint8)


#             masks = target['masks']
#             masks = masks.detach().cpu().numpy()

#             # store data.
#             gt_masks.append(masks)
#             pred_masks.append(masks_pred)
#             pred_scores.append(maskness_scores)

#         # masks2coco
#         self.gt_coco = masks2coco(gt_masks)
#         self.pred_coco = masks2coco(pred_masks, scores=pred_scores)
