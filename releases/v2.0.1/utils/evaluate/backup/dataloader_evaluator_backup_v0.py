import numpy as np
import torch
from torch import nn

from utils.coco.coco import COCO
from utils.coco.mask2coco import masks2coco
from pycocotools.cocoeval import COCOeval

from .coco_evaluator import Evaluator
from utils.utils import nested_tensor_from_tensor_list

from configs import cfg
import matplotlib.pyplot as plt

from utils.utils import flatten_mask

# coco_eval
class DataloaderEvaluator(Evaluator):
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
            target = batch[0]

            target = {k: v.to(cfg.device) for k, v in target.items()}
            images.append(target["image"])

            targets.append(target)
            images = nested_tensor_from_tensor_list(images)   # (B, C, H, W)

            # predict.
            output = self.inference_single(model, images.tensors)

            pred = output
            scores = pred['pred_logits'].sigmoid()
            scores = scores[0, :, 0]

            masks_pred = pred['pred_masks'].sigmoid()
            masks_pred = masks_pred[0, ...]
            
            # maskness scores.
            maskness_scores = []
            for p in masks_pred:
                maskness_score = torch.mean(p[p.gt(0.4)])
                maskness_score = torch.nan_to_num(maskness_score, nan=0.0)
                maskness_score = maskness_score.cpu()
                maskness_scores.append(maskness_score)

            maskness_scores = torch.tensor(maskness_scores).to(cfg.device)
            scores = maskness_scores

            # masks_pred = masks_pred[scores > 0.2]
            # scores = scores[scores > 0.2]

            scores = scores.detach().cpu().numpy()
            masks_pred = masks_pred.detach().cpu().numpy()

            masks_pred = (masks_pred > 0.4).astype(np.uint8)


            masks = target['masks']
            masks = masks.detach().cpu().numpy()

            # if step == 2:
            #     print(masks_pred.shape)
            #     print(np.min(masks_pred), np.max(masks_pred))
            #     # plt.figure()
            #     # plt.imshow(flatten_mask(masks_pred, 0)[0])
            #     # plt.savefig("./test.jpg")
            #     from utils.visualise import visualize_grid_v2

            #     visualize_grid_v2(
            #         masks=masks_pred, 
            #         titles=scores,
            #         ncols=10, 
            #         path=f'./test.jpg'
            #     )
            #     raise
            # else:
            #     continue

            # print(masks_pred.shape)
            # print(masks.shape)

            # store data.
            gt_masks.append(masks)
            pred_masks.append(masks_pred)
            pred_scores.append(scores)

        # masks2coco
        self.gt_coco = masks2coco(gt_masks)
        self.pred_coco = masks2coco(pred_masks, scores=pred_scores)
