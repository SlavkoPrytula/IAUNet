from typing import List, Dict, Any

import numpy as np
import torch
from torch import nn

from utils.coco.mask2coco import masks2coco, get_coco_template, create_image_info, create_annotation_info
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


@EVALUATORS.register(name="MemoryEfficientDataloaderEvaluator")
class MemoryEfficientDataloaderEvaluator(Evaluator):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.nms_threshold = cfg.nms_thr


    def add_annotations(self, masks: np.ndarray, image_id: int, annotation_id: int, is_ground_truth: bool, scores: List[float] = None):
        for index, mask in enumerate(masks):
            category_info = {'is_crowd': 0, 'id': 1}
            annotation_info = create_annotation_info(
                annotation_id=annotation_id + index,
                image_id=image_id,
                category_info=category_info,
                binary_mask=mask,
                image_size=mask.shape
            )
            if annotation_info is not None:
                if is_ground_truth:
                    self.gt_coco["annotations"].append(annotation_info)
                else:
                    if scores is not None:
                        annotation_info['score'] = scores[index]
                    else:  # <---
                        annotation_info['score'] = 1
                    # annotation_info['score'] = scores[index] if scores else 1
                    self.pred_coco["annotations"].append(annotation_info)
        return annotation_id + len(masks)


    def forward(self, model: torch.nn.Module, dataloader) -> None:
        model.eval()
        annotation_id = 1  # Initialize annotation ID

        start_time = time.time()  # Start time measurement
        start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # Memory in MB

        self.gt_coco = get_coco_template()
        self.pred_coco = get_coco_template()

        for image_id, batch in enumerate(dataloader):
            print(image_id)
            if batch is None:  # Skip empty batches
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
            seg_masks = masks_pred > self.mask_threshold
            sum_masks = seg_masks.sum((1, 2)).float()
            maskness_scores = (masks_pred * seg_masks.float()).sum((1, 2)) / sum_masks
            # scores = maskness_scores

            # masks_pred = masks_pred[scores > 0.4]
            # scores = scores[scores > 0.4]

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

            scores = maskness_scores


            scores = scores.detach().cpu().numpy()
            masks_pred = masks_pred.detach().cpu().numpy()
            masks_pred = (masks_pred > self.mask_threshold).astype(np.uint8)

            masks_gt = target['masks']
            masks_gt = masks_gt.detach().cpu().numpy()

            # Add ground truth annotations
            annotation_id = self.add_annotations(masks_gt, 
                                                 image_id, 
                                                 annotation_id, 
                                                 is_ground_truth=True)

            # Add predicted annotations
            annotation_id = self.add_annotations(masks_pred, 
                                                 image_id, 
                                                 annotation_id, 
                                                 is_ground_truth=False, 
                                                 scores=scores)

            image_info = create_image_info(
                image_id=image_id,
                image_size=masks_gt[0].shape,
                file_name=f"image{image_id}.jpg"
            )
            self.gt_coco["images"].append(image_info)
            self.pred_coco["images"].append(image_info)

        end_time = time.time()  # End time measurement
        end_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # Memory in MB

        time_elapsed = end_time - start_time
        memory_used = end_memory - start_memory  # Calculate additional memory consumed

        print(f"Processing Time: {time_elapsed:.2f} seconds")
        print(f"Memory Used: {memory_used:.2f} MB")
