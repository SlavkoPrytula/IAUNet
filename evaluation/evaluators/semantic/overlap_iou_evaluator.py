import torch 
from tqdm import tqdm
from utils.utils import nested_tensor_from_tensor_list
from utils.opt.mask_nms import mask_nms
import numpy as np

from configs import cfg
from ..base_evaluator import BaseEvaluator

from utils.registry import EVALUATORS
from utils.common.decorators import timeit_evaluator, memory_evaluator

from ...metrics.iou import IOUMetric


@timeit_evaluator
@memory_evaluator
@EVALUATORS.register(name="OverlapIOUEvaluator")
class OverlapIOUEvaluator(BaseEvaluator):
    def __init__(self, cfg: cfg, model=None, dataset=None, **kwargs):
        super().__init__(cfg, model, **kwargs)
        self.score_threshold = cfg.model.evaluator.score_thr
        self.mask_threshold = cfg.model.evaluator.mask_thr
        self.nms_threshold = cfg.model.evaluator.nms_thr

        self.dataset = dataset
        self.num_classes = cfg.model.decoder.instance_head.num_classes
        self.iou_metric = IOUMetric()
        self.stats = {"mean_IoU": 0}


    def forward(self, dataloader):
        super().forward(dataloader)

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), miniters=5)
        for step, batch in pbar:
            if batch is None:
                continue
            
            # prepare targets
            images, targets = [], []
            for i in range(len(batch)):
                target = batch[i]

                ignore = ["img_id", "img_path", "ori_shape", "file_name", "coco_id"]
                target = {k: v.to(self.device) if k not in ignore else v 
                        for k, v in target.items()}
                images.append(target["image"])
                targets.append(target)

            image = nested_tensor_from_tensor_list(images)

            # ============= PREDICTION ==============
            # predict.
            preds = self.inference_single(image.tensors)
            preds["img_id"] = [targets[i]["img_id"] for i in range(len(targets))]
            preds["ori_shape"] = [targets[i]["ori_shape"] for i in range(len(targets))]
            preds["instance_masks"] = [targets[i]["instance_masks"] for i in range(len(targets))]

            self.process(preds)
    

    def process(self, preds: dict):
        scores_batch = preds['pred_logits'].softmax(-1)
        masks_pred_batch = preds['pred_instance_masks'].sigmoid()
        iou_scores_batch = preds['pred_scores'].sigmoid()
        bboxes_pred_batch = preds['pred_bboxes']
        masks_gt_batch = preds['instance_masks']

        for batch_idx, (scores, masks_pred, iou_scores, bboxes_pred, masks_gt) in enumerate(zip(
            scores_batch, masks_pred_batch, iou_scores_batch, bboxes_pred_batch, masks_gt_batch)):
            scores = scores[:, :-1]
            iou_scores = iou_scores.flatten(0, 1)

            labels = torch.arange(self.num_classes, device=scores.device).unsqueeze(0).repeat(masks_pred.shape[0], 1).flatten(0, 1)
            scores, topk_indices = scores.flatten(0, 1).topk(masks_pred.shape[0], sorted=False)
            labels = labels[topk_indices]

            topk_indices = topk_indices // self.num_classes
            masks_pred = masks_pred[topk_indices]
            iou_scores = iou_scores[topk_indices]
            bboxes_pred = bboxes_pred[topk_indices]


            # maskness scores.
            seg_masks = masks_pred > self.mask_threshold
            sum_masks = seg_masks.sum((1, 2)).float()
            maskness_scores = (masks_pred * seg_masks.float()).sum((1, 2)) / (sum_masks + 1e-6)
            
            # scores = torch.sqrt(scores * iou_scores)
            scores = scores * maskness_scores

            # ========== CLS Score ==========
            # # score filtering.
            keep = scores > self.score_threshold
            masks_pred = masks_pred[keep]
            scores = scores[keep]
            labels = labels[keep]
            iou_scores = iou_scores[keep]
            bboxes_pred = bboxes_pred[keep]

            # ========== NMS ==========
            # pre_nms sort.
            sort_inds = torch.argsort(scores, descending=True)
            masks_pred = masks_pred[sort_inds, :, :]
            scores = scores[sort_inds]
            labels = labels[sort_inds]
            iou_scores = iou_scores[sort_inds]
            bboxes_pred = bboxes_pred[sort_inds]

            # nms.
            seg_masks = masks_pred > self.mask_threshold
            sum_masks = seg_masks.sum((1, 2)).float()
            
            keep = mask_nms(labels, seg_masks, sum_masks, scores, nms_thr=self.nms_threshold)
            masks_pred = masks_pred[keep, :, :]
            scores = scores[keep]
            labels = labels[keep]
            iou_scores = iou_scores[keep]
            bboxes_pred = bboxes_pred[keep]

            masks_pred = masks_pred > self.mask_threshold
            # ================================================

            masks_overlap_pred = self.get_overlaps(masks_pred)
            masks_overlap_gt = self.get_overlaps(masks_gt)

            results = {
                "pred_masks": masks_overlap_pred,
                "gt_masks": masks_overlap_gt,
            }

            self.iou_metric.process({}, [results])
    

    def evaluate(self, verbose=False):
        size = len(self.dataset)
        eval_results = self.iou_metric.evaluate(size)

        if verbose:
            print(f"mean overlaps iou: {eval_results['mean_IoU']:.4f}")

        for key, value in eval_results.items():
            self.stats[key] = value


    # def get_overlaps(self, masks):
    #     _, _, N = masks.shape
    #     overlaps = torch.zeros_like(masks, dtype=torch.bool)
    #     all_masks_summed = masks.sum(dim=-1)

    #     for i in range(N):
    #         overlaps[:, :, i] = torch.logical_and(masks[:, :, i], (all_masks_summed - masks[:, :, i]) > 0)
        
    #     return overlaps

    def get_overlaps(self, masks):
        all_masks_summed = masks.sum(dim=0)
        overlaps = (all_masks_summed > 1).to(torch.bool)
        overlaps = overlaps.float()

        return overlaps




