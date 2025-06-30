import torch
from os.path import join
from configs import cfg
import torch.nn.functional as F

from .base_evaluator import BaseEvaluator
from utils.registry import EVALUATORS
from evaluation.mmdet import CocoMetric
from utils.box_ops import box_cxcywh_to_xyxy


# def remove_padding(mask, ori_shape, rescale=False):
#         mask_h, mask_w = mask.shape[-2:]
#         ori_h, ori_w = ori_shape
        
#         scale = min(mask_h / ori_h, mask_w / ori_w)
        
#         new_h = int(ori_h * scale)
#         new_w = int(ori_w * scale)
        
#         pad_top = (mask_h - new_h) // 2
#         pad_left = (mask_w - new_w) // 2
        
#         mask = mask[:, pad_top:new_h + pad_top, pad_left:new_w + pad_left]

#         if rescale:
#             mask = F.interpolate(mask.float().unsqueeze(0), size=ori_shape, 
#                                 mode="bilinear", align_corners=False).squeeze(0)
        
#         return mask


def remove_padding(mask, img_size, output_height, output_width, rescale=False):
        mask = mask[:, :img_size[0], :img_size[1]]

        if rescale:
            mask = F.interpolate(mask.float().unsqueeze(0), size=(output_height, output_width), 
                                mode="bilinear", align_corners=False).squeeze(0)
        
        return mask


@EVALUATORS.register(name="CocoEvaluator")
class CocoEvaluator(BaseEvaluator):
    # coco_eval
    def __init__(self, cfg: cfg, dataset=None, **kwargs):
        super().__init__(cfg, **kwargs)

        self.dataset = dataset
        outfile_prefix = join(cfg.run.save_dir, cfg.model.evaluator.outfile_prefix) \
            if (cfg.model.evaluator.outfile_prefix and cfg.run.get("save_dir")) else None
        classwise = cfg.model.evaluator.classwise
        metric = cfg.model.evaluator.metric \
            if isinstance(cfg.model.evaluator.metric, str) else list(cfg.model.evaluator.metric)
        self.num_classes = cfg.model.decoder.num_classes

        print(f"Doing evaluation on dataset.ann_file: {dataset.ann_file}")

        self.metric = CocoMetric(
            ann_file=dataset.ann_file,
            metric=metric,
            classwise=classwise,
            outfile_prefix=outfile_prefix,
            )

        categories = self.metric._coco_api.loadCats(self.metric._coco_api.getCatIds())
        class_names = [category['name'] for category in categories]
        self.metric.dataset_meta = dict(classes=class_names)

        self.score_threshold = cfg.model.evaluator.score_thr
        self.mask_threshold = cfg.model.evaluator.mask_thr
        self.nms_threshold = cfg.model.evaluator.nms_thr


    def process(self, preds: dict):
        # TODO: this should be handled in the model predicition head.
        scores_batch = preds['pred_logits'].softmax(-1)
        # scores_batch = preds['pred_logits'].sigmoid()
        masks_pred_batch = preds['pred_instance_masks'].sigmoid()
        bboxes_pred_batch = preds['pred_bboxes']

        for batch_idx, (scores, masks_pred, bboxes_pred) in enumerate(zip(
            scores_batch, masks_pred_batch, bboxes_pred_batch)):
            ori_shape = preds["ori_shape"][batch_idx]

            scores = scores[:, :-1]

            # postprocessing.
            if masks_pred.shape[0]:
                masks_pred = remove_padding(
                    masks_pred,
                    img_size=preds['resized_shape'][batch_idx],
                    output_height=ori_shape[0],
                    output_width=ori_shape[1],
                    rescale=True
                )

            labels = torch.arange(self.num_classes, device=scores.device).unsqueeze(0).repeat(masks_pred.shape[0], 1).flatten(0, 1)
            scores, topk_indices = scores.flatten(0, 1).topk(masks_pred.shape[0], sorted=False)
            labels = labels[topk_indices]

            topk_indices = topk_indices // self.num_classes
            masks_pred = masks_pred[topk_indices]
            bboxes_pred = bboxes_pred[topk_indices]

            img_h, img_w = ori_shape
            bboxes_pred = box_cxcywh_to_xyxy(bboxes_pred)
            bboxes_pred = bboxes_pred * torch.tensor([img_h, img_w, img_h, img_w], dtype=torch.float32, device=masks_pred.device)

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
            # sort_inds = torch.argsort(scores, descending=True)
            # masks_pred = masks_pred[sort_inds]
            # scores = scores[sort_inds]
            # labels = labels[sort_inds]
            # bboxes_pred = bboxes_pred[sort_inds]

            # # nms.
            # seg_masks = masks_pred > self.mask_threshold
            # sum_masks = seg_masks.sum((1, 2)).float()
            
            # keep = mask_nms(labels, seg_masks, sum_masks, scores, nms_thr=self.nms_threshold)
            # masks_pred = masks_pred[keep]
            # scores = scores[keep]
            # labels = labels[keep]
            # # iou_scores = iou_scores[keep]
            # bboxes_pred = bboxes_pred[keep]

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
            # segm metrics
            'coco/segm_mAP': 'segm_mAP',
            'coco/segm_mAP_50': 'segm_mAP_50',
            'coco/segm_mAP_75': 'segm_mAP_75',
            'coco/segm_mAP_s': 'segm_mAP_s',
            'coco/segm_mAP_m': 'segm_mAP_m',
            'coco/segm_mAP_l': 'segm_mAP_l',
            # bbox metrics
            'coco/bbox_mAP': 'bbox_mAP',
            'coco/bbox_mAP_50': 'bbox_mAP_50',
            'coco/bbox_mAP_75': 'bbox_mAP_75',
            'coco/bbox_mAP_s': 'bbox_mAP_s',
            'coco/bbox_mAP_m': 'bbox_mAP_m',
            'coco/bbox_mAP_l': 'bbox_mAP_l',
        }

        # Compute metrics
        size = len(self.dataset)
        eval_results = self.metric.evaluate(size)

        # Update self.stats based on the mapping
        self.stats = {}
        for key, value in eval_results.items():
            if key in key_mapping:
                value if value != -1 else 0
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