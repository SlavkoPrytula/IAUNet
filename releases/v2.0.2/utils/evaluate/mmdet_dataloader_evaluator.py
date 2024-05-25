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

from utils.registry import EVALUATORS, DATASETS
from evaluation.mmdet import CocoMetric


import time
import psutil
import os


@EVALUATORS.register(name="MMDetDataloaderEvaluator")
class MMDetDataloaderEvaluator(Evaluator):
    # coco_eval
    def __init__(self, cfg: cfg, dataset=None):
        super(MMDetDataloaderEvaluator, self).__init__(cfg)

        self.coco_metric = CocoMetric(
            ann_file=dataset.ann_file, 
            metric='segm', 
            classwise=False,
            outfile_prefix=f"{cfg.save_dir}/results/coco"
            )

        categories = self.coco_metric._coco_api.loadCats(self.coco_metric._coco_api.getCatIds())
        class_names = [category['name'] for category in categories]
        self.coco_metric.dataset_meta = dict(classes=class_names)

        self.nms_threshold = cfg.model.evaluator.nms_thr


    def forward(self, model, dataloader):
        model.eval()

        start_time = time.time()  # Start time measurement
        start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # Memory in MB

        for step, batch in enumerate(dataloader):
            if batch is None:
                continue
            
            # prepare targets
            images = []
            targets = []

            target = batch[0]
            ignore = ["img_id", "img_path", "ori_shape"]
            target = {k: v.to(cfg.device) if k not in ignore else v 
                    for k, v in target.items()}
            images.append(target["image"])
            targets.append(target)

            image = nested_tensor_from_tensor_list(images)


            # ============= PREDICTION ==============
            # predict.
            pred = self.inference_single(model, image.tensors)

            scores = pred['pred_logits'].softmax(-1)
            masks_pred = pred['pred_masks'].sigmoid()
            
            masks_pred = masks_pred[0, ...]
            scores = scores[0, :, :-1]
            # scores = scores[0, :, 0]
            # labels = scores.argmax(-1)
            # labels = torch.zeros(len(scores), dtype=torch.int64)


            labels = torch.arange(cfg.model.num_classes, device=scores.device).unsqueeze(0).repeat(cfg.model.num_masks, 1).flatten(0, 1)
            scores, topk_indices = scores.flatten(0, 1).topk(100, sorted=False)
            labels = labels[topk_indices]

            topk_indices = topk_indices // cfg.model.num_classes
            masks_pred = masks_pred[topk_indices]
            print(scores)
            print(f"num_preds after top_k: {len(scores)}")


            # maskness scores.
            seg_masks = masks_pred > self.mask_threshold
            sum_masks = seg_masks.sum((1, 2)).float()
            maskness_scores = (masks_pred * seg_masks.float()).sum((1, 2)) / sum_masks
            scores = scores * maskness_scores

            # ========== CLS Score ==========
            # # score filtering.
            # keep = scores > self.score_threshold
            # masks_pred = masks_pred[keep]
            # scores = scores[keep]
            # labels = labels[keep]
            # print(f"num_preds after 1st cls_thr: {len(scores)}")


            # ========== NMS ==========
            # # pre_nms sort.
            # sort_inds = torch.argsort(scores, descending=True)
            # masks_pred = masks_pred[sort_inds, :, :]
            # scores = scores[sort_inds]
            # labels = labels[sort_inds]

            # # nms.
            # seg_masks = masks_pred > self.mask_threshold
            # sum_masks = seg_masks.sum((1, 2)).float()
            
            # keep = mask_nms(labels, seg_masks, sum_masks, scores, nms_thr=self.nms_threshold)
            # masks_pred = masks_pred[keep, :, :]
            # scores = scores[keep]
            # labels = labels[keep]
            # print(scores)
            # print(f"num_preds after nms: {len(scores)}")



            # # NEW: moved up before mask rescoring
            # print(f"raw num preds:      {len(scores)}")
            # print(scores)
            # keep = scores > self.score_threshold
            # masks_pred = masks_pred[keep]
            # scores = scores[keep]
            # labels = labels[keep]
            # print(f"num_preds after 1st cls_thr: {len(scores)}")
            # print(f"filtered num preds: {len(scores)}")


            masks_pred = masks_pred > self.mask_threshold
            # ================================================


            results = dict()
            results["img_id"] = target["img_id"]
            results["ori_shape"] = target["ori_shape"]
            results["pred_instances"] = {
                "masks": masks_pred,
                "labels": labels,
                "scores": scores,
                "mask_scores": scores,
                "bboxes": torch.zeros(len(scores), 4),
            }

            data_samples = [results]
            self.coco_metric.process({}, data_samples)



        end_time = time.time()  # End time measurement
        end_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # Memory in MB

        time_elapsed = end_time - start_time
        memory_used = end_memory - start_memory  # Calculate additional memory consumed

        print(f"Processing Time: {time_elapsed:.2f} seconds")
        print(f"Memory Used: {memory_used:.2f} MB")


    def evaluate(self, verbose=False):

        key_mapping = {
            'segm_mAP': "mAP@0.5:0.95",
            'segm_mAP_50': "mAP@0.5",
            'segm_mAP_75': "mAP@0.75",
            'segm_mAP_s': "mAP(s)@0.5",
            'segm_mAP_m': "mAP(m)@0.5",
            'segm_mAP_l': "mAP(l)@0.5",
        }

         # Compute metrics
        eval_results = self.coco_metric.evaluate()

        # Update self.stats based on the mapping
        for key, value in eval_results.items():
            if key in key_mapping:
                self.stats[key_mapping[key]] = value

        self.gt_coco = self.coco_metric._coco_api
        self.pred_coco = self.coco_metric.coco_dt
