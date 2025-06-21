import torch
from os.path import join
from configs import cfg
from tqdm import tqdm
import torch.nn.functional as F

from .coco_evaluator import COCOEvaluator
from utils.utils import nested_tensor_from_tensor_list
from utils.opt.mask_nms import mask_nms
from utils.box_ops import box_cxcywh_to_xyxy
from utils.registry import EVALUATORS, DATASETS
from evaluation.mmdet import CocoMetric

from utils.common.decorators import timeit_evaluator, memory_evaluator


def remove_padding(mask, ori_shape, rescale=False):
        mask_h, mask_w = mask.shape[-2:]
        ori_h, ori_w = ori_shape
        
        scale = min(mask_h / ori_h, mask_w / ori_w)
        
        new_h = int(ori_h * scale)
        new_w = int(ori_w * scale)
        
        pad_top = (mask_h - new_h) // 2
        pad_left = (mask_w - new_w) // 2
        
        mask = mask[:, pad_top:new_h + pad_top, pad_left:new_w + pad_left]

        if rescale:
            mask = F.interpolate(mask.float().unsqueeze(0), size=ori_shape, 
                                mode="bilinear", align_corners=False).squeeze(0)
        
        return mask


@timeit_evaluator
@memory_evaluator
@EVALUATORS.register(name="MMDetDataloaderEvaluator")
class MMDetDataloaderEvaluator(COCOEvaluator):
    # coco_eval
    def __init__(self, cfg: cfg, dataset=None, **kwargs):
        super().__init__(cfg, **kwargs)

        self.dataset = dataset
        outfile_prefix = cfg.model.evaluator.outfile_prefix
        self.num_classes = cfg.model.decoder.num_classes
        self.metric = cfg.model.evaluator.metric

        print(f"Doing evaluation on dataset.ann_file: {dataset.ann_file}")

        self.metric = CocoMetric(
            ann_file=dataset.ann_file,
            metric=cfg.model.evaluator.metric,
            classwise=cfg.model.evaluator.classwise,
            outfile_prefix=join(cfg.run.save_dir, outfile_prefix) if (outfile_prefix and cfg.run.get("save_dir")) else None,
            )

        categories = self.metric._coco_api.loadCats(self.metric._coco_api.getCatIds())
        class_names = [category['name'] for category in categories]
        self.metric.dataset_meta = dict(classes=class_names)

        self.score_threshold = cfg.model.evaluator.score_thr
        self.mask_threshold = cfg.model.evaluator.mask_thr
        self.nms_threshold = cfg.model.evaluator.nms_thr


    def process(self, preds: dict):
        scores_batch = preds['pred_logits'].softmax(-1)
        masks_pred_batch = preds['pred_instance_masks'].sigmoid()
        bboxes_pred_batch = preds['pred_bboxes']

        for batch_idx, (scores, masks_pred, bboxes_pred) in enumerate(zip(
            scores_batch, masks_pred_batch, bboxes_pred_batch)):
            scores = scores[:, :-1]

            labels = torch.arange(self.num_classes, device=scores.device).unsqueeze(0).repeat(masks_pred.shape[0], 1).flatten(0, 1)
            scores, topk_indices = scores.flatten(0, 1).topk(masks_pred.shape[0], sorted=False)
            labels = labels[topk_indices]

            topk_indices = topk_indices // self.num_classes
            masks_pred = masks_pred[topk_indices]
            bboxes_pred = bboxes_pred[topk_indices]


            # maskness scores.
            seg_masks = masks_pred > self.mask_threshold
            sum_masks = seg_masks.sum((1, 2)).float()
            maskness_scores = (masks_pred * seg_masks.float()).sum((1, 2)) / (sum_masks + 1e-6)
            scores = scores * maskness_scores

            # ========== CLS Score ==========
            # score filtering.
            keep = scores > self.score_threshold
            masks_pred = masks_pred[keep]
            scores = scores[keep]
            labels = labels[keep]
            bboxes_pred = bboxes_pred[keep]

            # ========== NMS ==========
            # pre_nms sort.
            sort_inds = torch.argsort(scores, descending=True)
            masks_pred = masks_pred[sort_inds]
            scores = scores[sort_inds]
            labels = labels[sort_inds]
            bboxes_pred = bboxes_pred[sort_inds]

            # # nms.
            # seg_masks = masks_pred > self.mask_threshold
            # sum_masks = seg_masks.sum((1, 2)).float()
            
            # keep = mask_nms(labels, seg_masks, sum_masks, scores, nms_thr=self.nms_threshold)
            # masks_pred = masks_pred[keep]
            # scores = scores[keep]
            # labels = labels[keep]
            # # iou_scores = iou_scores[keep]
            # bboxes_pred = bboxes_pred[keep]

            # postprocessing - currently done here, should be moved to model.
            ori_shape = preds["ori_shape"][batch_idx]
            if masks_pred.shape[0]:
                masks_pred = remove_padding(
                    masks_pred, 
                    ori_shape,
                    rescale=True
                )

            masks_pred = masks_pred > self.mask_threshold
            # ================================================

            results = dict()
            results["img_id"] = preds["img_id"][batch_idx]
            results["ori_shape"] = preds["ori_shape"][batch_idx]
            results["pred_instances"] = {
                "masks": masks_pred,
                "labels": labels,
                "scores": scores,
                "mask_scores": scores,
                "bboxes": bboxes_pred,
            }

            data_samples = [results]
            self.metric.process({}, data_samples)


    def evaluate(self, verbose=False):
        key_mapping = {
            'coco/segm_mAP': "mAP@0.5:0.95",
            'coco/segm_mAP_50': "mAP@0.5",
            'coco/segm_mAP_75': "mAP@0.75",
            'coco/segm_mAP_s': "mAP(s)@0.5",
            'coco/segm_mAP_m': "mAP(m)@0.5",
            'coco/segm_mAP_l': "mAP(l)@0.5",
        }

        # Compute metrics
        size = len(self.dataset)
        eval_results = self.metric.evaluate(size)

        # Update self.stats based on the mapping
        for key, value in eval_results.items():
            if key in key_mapping:
                self.stats[key_mapping[key]] = value

        self.gt_coco = self.metric._coco_api
        self.pred_coco = self.metric.coco_dt


    def __repr__(self):
        head = "Evaluator " + self.__class__.__name__
        body = [
            f"dataset: {self.dataset.ann_file}",
            f"num_classes: {self.num_classes}",
            f"mask_threshold: {self.mask_threshold}",
            f"score_threshold: {self.score_threshold}",
            f"nms_threshold: {self.nms_threshold}",
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]

        return "\n" + "\n".join(lines) + "\n"