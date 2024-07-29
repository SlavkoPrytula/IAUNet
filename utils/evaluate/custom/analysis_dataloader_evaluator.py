import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# from utils.coco.coco import COCO
from utils.coco.mask2coco import masks2coco
# from pycocotools.cocoeval import COCOeval

from ..mmdet_dataloader_evaluator import MMDetDataloaderEvaluator
from utils.utils import nested_tensor_from_tensor_list

from configs import cfg
import matplotlib.pyplot as plt

from utils.utils import flatten_mask
from utils.opt.mask_nms import mask_nms

from utils.registry import EVALUATORS, DATASETS, build_criterion
from evaluation.mmdet import CocoMetric

from utils.utils import compute_mask_iou
from scipy.stats import pearsonr



@EVALUATORS.register(name="AnalysisDataloaderEvaluator")
class AnalysisDataloaderEvaluator(MMDetDataloaderEvaluator):
    def __init__(self, cfg: cfg, dataset=None):
        super().__init__(cfg, dataset)
        self.criterion = build_criterion(cfg=cfg.model.criterion)

    @torch.no_grad()
    def forward(self, model, dataloader):
        model.eval()

        all_scores = []
        all_ious = []

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
            pred = model(image.tensors)
            _, (src_idx, tgt_idx) = self.criterion(pred, targets, [512, 512], return_matches=True, epoch=None)

            # match gt and preds.
            indices = src_idx[0] == 0
            src_idx = src_idx[1][indices] # [1, 0, 2, 3 ...]

            indices = tgt_idx[0] == 0
            tgt_idx = tgt_idx[1][indices] # [3, 1, 2, 5 ...]

            
            masks1 = pred['pred_masks'][0][src_idx]  # (M, H, W)
            masks2 = targets[0]["masks"][tgt_idx]    # (M, H, W)

            masks1 = masks1.sigmoid()

            masks1 = masks1.flatten(1)
            masks2 = masks2.flatten(1)
            ious = compute_mask_iou(masks1, masks2)


            scores = pred['pred_logits'].softmax(-1)
            iou_scores = pred['pred_scores'].sigmoid()
            masks_pred = pred['pred_masks'].sigmoid()
            
            masks_pred = masks_pred[0, ...][src_idx]
            scores = scores[0, :, :-1][src_idx]
            iou_scores = iou_scores[0, ..., 0][src_idx]
            # scores = scores * iou_scores
            # scores = iou_scores


            labels = torch.arange(cfg.model.num_classes, device=scores.device).unsqueeze(0).repeat(cfg.model.num_masks, 1).flatten(0, 1)
            scores, topk_indices = scores.flatten(0, 1).topk(len(scores), sorted=False)
            labels = labels[topk_indices]

            topk_indices = topk_indices // cfg.model.num_classes
            masks_pred = masks_pred[topk_indices]
            print(f"num_preds after top_k: {len(scores)}")



            # maskness scores.
            seg_masks = masks_pred > self.mask_threshold
            sum_masks = seg_masks.sum((1, 2)).float()
            sum_masks = sum_masks.clamp(min=1e-9)
            maskness_scores = (masks_pred * seg_masks.float()).sum((1, 2)) / sum_masks
            # scores = scores * maskness_scores
            # scores = maskness_scores


            # score filtering.
            keep = scores > self.score_threshold
            masks_pred = masks_pred[keep]
            scores = scores[keep]
            labels = labels[keep]

            ious = ious[keep]
            maskness_scores = maskness_scores[keep]
            iou_scores = iou_scores[keep]
            print(f"num_preds after 1st cls_thr: {len(scores)}")



            # pre_nms sort.
            sort_inds = torch.argsort(scores, descending=True)
            masks_pred = masks_pred[sort_inds, :, :]
            scores = scores[sort_inds]
            labels = labels[sort_inds]
            
            ious = ious[sort_inds]
            maskness_scores = maskness_scores[sort_inds]
            iou_scores = iou_scores[sort_inds]


            # nms.
            seg_masks = masks_pred > self.mask_threshold
            sum_masks = seg_masks.sum((1, 2)).float()
            
            keep = mask_nms(labels, seg_masks, sum_masks, scores, nms_thr=self.nms_threshold)
            masks_pred = masks_pred[keep, :, :]
            scores = scores[keep]
            labels = labels[keep]
            # print(scores)

            ious = ious[keep]
            maskness_scores = maskness_scores[keep]
            iou_scores = iou_scores[keep]
            print(f"num_preds after nms: {len(scores)}")


            # all_scores.extend(scores.cpu().detach().numpy())
            all_scores.extend(ious.cpu().detach().numpy())
            # all_scores.extend(iou_scores.cpu().detach().numpy())
            # all_ious.extend(ious.cpu().detach().numpy())
            # all_ious.extend(maskness_scores.cpu().detach().numpy())
            all_ious.extend(iou_scores.cpu().detach().numpy())


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


        plt.figure(figsize=[10, 10])
        plt.scatter(all_scores, all_ious, 
                    color="orange", s=200, alpha=0.5)
        
        correlation, _ = pearsonr(all_scores, all_ious)
        print(correlation)
        
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        # plt.xlabel("Cls (rescored + filter + nms)", fontsize=24)
        # plt.xlabel("Cls (filter + nms)", fontsize=24)
        # plt.xlabel("Cls", fontsize=24)
        plt.xlabel("IOU", fontsize=24)
        plt.ylabel("Pred IOU", fontsize=24)
        # plt.xlabel("Maskness", fontsize=24)
        # plt.ylabel("Maskness", fontsize=24)
        # plt.ylabel("IOU", fontsize=24)
        plt.title(f"Pearson: {correlation:.2f}", fontsize=24)
        plt.grid(True, alpha=0.75)
        plt.tight_layout()
        plt.savefig("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/utils/evaluate/temp.jpg")
