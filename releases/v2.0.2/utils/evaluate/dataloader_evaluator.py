import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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


import time
import psutil
import os


# TODO: merge base and nms evaluators
@EVALUATORS.register(name="DataloaderEvaluator")
class DataloaderEvaluator(Evaluator):
    # coco_eval
    def __init__(self, cfg: cfg):
        super(DataloaderEvaluator, self).__init__(cfg)

    def forward(self, model, dataloader):
        model.eval()

        start_time = time.time()  # Start time measurement
        start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # Memory in MB


        gt_masks = []
        pred_masks = []
        pred_scores = []

        for step, batch in enumerate(dataloader):
            if batch is None:
                continue
            
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
            pred = self.inference_single(model, image.tensors)

            # scores = pred['pred_logits'].sigmoid()
            # scores, labels = F.softmax(output['pred_logits'], dim=-1).max(-1)
            # scores = scores[0, :]
            # print(f"de: {scores.shape}")
            # print(scores)

            scores = pred['pred_logits'].softmax(-1)
            masks_pred = pred['pred_masks'].sigmoid()
            
            masks_pred = masks_pred[0, ...]
            scores = scores[0, :, :-1]


            labels = torch.arange(cfg.model.num_classes, device=scores.device).unsqueeze(0).repeat(cfg.model.num_masks, 1).flatten(0, 1)

            scores, topk_indices = scores.flatten(0, 1).topk(50, sorted=False)
            print(scores)
            labels = labels[topk_indices]

            topk_indices = topk_indices // cfg.model.num_classes
            masks_pred = masks_pred[topk_indices]


            # NEW: moved up before mask rescoring
            # print(f"raw num preds:      {len(scores)}")
            # masks_pred = masks_pred[scores > self.score_threshold]
            # scores = scores[scores > self.score_threshold]
            print(f"filtered num preds: {len(scores)}")



            # maskness scores.
            seg_masks = masks_pred > self.mask_threshold
            sum_masks = seg_masks.sum((1, 2)).float()
            maskness_scores = (masks_pred * seg_masks.float()).sum((1, 2)) / sum_masks
            
            
            scores = scores * maskness_scores


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
        try:
            self.gt_coco = masks2coco(gt_masks)
            self.pred_coco = masks2coco(pred_masks, scores=pred_scores)
        except:
            print("WARNING: no predictions found!")

        end_time = time.time()  # End time measurement
        end_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # Memory in MB

        time_elapsed = end_time - start_time
        memory_used = end_memory - start_memory  # Calculate additional memory consumed

        print(f"Processing Time: {time_elapsed:.2f} seconds")
        print(f"Memory Used: {memory_used:.2f} MB")



@EVALUATORS.register(name="DataloaderEvaluatorNMS")
class DataloaderEvaluatorNMS(Evaluator):
    # coco_eval - nms
    def __init__(self, cfg: cfg):
        super(DataloaderEvaluatorNMS, self).__init__(cfg)

        self.nms_threshold = cfg.nms_thr

    def forward(self, model, dataloader):
        model.eval()
        
        gt_masks = []
        pred_masks = []
        pred_scores = []

        for step, batch in enumerate(dataloader):
            print(step)
            if batch is None:
                continue  
            
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

            N, H, W = masks_pred.shape
            
            # maskness scores.
            # maskness_scores = []
            # for p in masks_pred:
            #     maskness_score = torch.mean(p[p.gt(self.mask_threshold)])
            #     maskness_score = torch.nan_to_num(maskness_score, nan=0.0)
            #     maskness_score = maskness_score.cpu()
            #     maskness_scores.append(maskness_score)

            # maskness_scores = torch.tensor(maskness_scores).to(cfg.device)
            # scores = maskness_scores

            # masks_pred = masks_pred[scores > 0.2]
            # scores = scores[scores > 0.2]

            # scores = scores.detach().cpu().numpy()
            # masks_pred = masks_pred.detach().cpu().numpy()
            # masks_pred = (masks_pred > self.mask_threshold).astype(np.uint8)

            # keep = scores > 0.5
            # masks_pred = masks_pred[keep]
            # scores = scores[keep]

            N, H, W = masks_pred.shape

            seg_masks = masks_pred > self.mask_threshold
            sum_masks = seg_masks.sum((1, 2)).float()

            # maskness scores.
            maskness_scores = (masks_pred * seg_masks.float()).sum((1, 2)) / sum_masks

            # print(maskness_scores)
            # print(scores)
            # print(zip(maskness_scores, scores))
            # maskness_scores = np.ones(N)

            # maskness_scores = scores ** (1-maskness_scores)

            # sort predictions
            sort_inds = torch.argsort(maskness_scores, descending=True)
            seg_masks = seg_masks[sort_inds, :, :]
            masks_pred = masks_pred[sort_inds, :, :]
            sum_masks = sum_masks[sort_inds]
            maskness_scores = maskness_scores[sort_inds]
            scores = scores[sort_inds]
            labels = torch.ones(N)

            # nms
            keep = mask_nms(labels, seg_masks, sum_masks, maskness_scores, nms_thr=self.nms_threshold)
            masks_pred = masks_pred[keep, :, :]
            maskness_scores = maskness_scores[keep]
            scores = scores[keep]

            maskness_scores = maskness_scores.detach().cpu().numpy()

            masks_pred = masks_pred.detach().cpu().numpy()
            masks_pred = (masks_pred > self.mask_threshold).astype(np.uint8)
            
            keep = maskness_scores > self.score_threshold
            masks_pred = masks_pred[keep]
            maskness_scores = maskness_scores[keep]

            masks = target['masks']
            masks = masks.detach().cpu().numpy()

            # store data.
            gt_masks.append(masks)
            pred_masks.append(masks_pred)
            pred_scores.append(maskness_scores)

        # masks2coco
        self.gt_coco = masks2coco(gt_masks)
        self.pred_coco = masks2coco(pred_masks, scores=pred_scores)
