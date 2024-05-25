import torch
from os.path import join
from configs import cfg
from tqdm import tqdm

from .coco_evaluator import Evaluator
from utils.utils import nested_tensor_from_tensor_list
from utils.opt.mask_nms import mask_nms
from utils.registry import EVALUATORS, DATASETS
from evaluation.mmdet import CocoMetric

from utils.common.decorators import timeit_evaluator, memory_evaluator


@timeit_evaluator
@memory_evaluator
@EVALUATORS.register(name="MMDetDataloaderEvaluator")
class MMDetDataloaderEvaluator(Evaluator):
    # coco_eval
    def __init__(self, cfg: cfg, model=None, dataset=None, **kwargs):
        super(MMDetDataloaderEvaluator, self).__init__(cfg, model, **kwargs)

        outfile_prefix = cfg.model.evaluator.outfile_prefix
        coco_api = cfg.model.evaluator.coco_api
        self.num_classes = cfg.model.instance_head.num_classes

        print(f"dataset.ann_file: {dataset.ann_file}")

        self.coco_metric = CocoMetric(
            ann_file=dataset.ann_file,
            metric=cfg.model.evaluator.metric,
            classwise=cfg.model.evaluator.classwise,
            outfile_prefix=join(cfg.save_dir, outfile_prefix) if (outfile_prefix and hasattr(cfg, 'save_dir')) else None,
            coco_api=coco_api if coco_api else 'COCOeval'
            )

        categories = self.coco_metric._coco_api.loadCats(self.coco_metric._coco_api.getCatIds())
        class_names = [category['name'] for category in categories]
        self.coco_metric.dataset_meta = dict(classes=class_names)

        self.nms_threshold = cfg.model.evaluator.nms_thr


    def forward(self, dataloader):
        super().forward(dataloader)

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), miniters=5)
        for step, batch in pbar:
            if batch is None:
                continue
            
            # prepare targets
            images = []
            targets = []

            target = batch[0]
            ignore = ["img_id", "img_path", "ori_shape", "file_name", "coco_id"]
            target = {k: v.to(cfg.device) if k not in ignore else v 
                    for k, v in target.items()}
            images.append(target["image"])
            targets.append(target)

            image = nested_tensor_from_tensor_list(images)


            # ============= PREDICTION ==============
            # predict.
            pred = self.inference_single(image.tensors)

            pred["img_id"] = target["img_id"]
            pred["ori_shape"] = target["ori_shape"]

            self.process(pred)

    
    def process(self, pred: dict):
        scores = pred['pred_logits'].softmax(-1)
        masks_pred = pred['pred_masks'].sigmoid()
        iou_scores = pred['pred_scores'].sigmoid()
        bboxes_pred = pred['pred_bboxes']
        
        masks_pred = masks_pred[0, ...]
        scores = scores[0, :, :-1]
        iou_scores = iou_scores[0, ...].flatten(0, 1)
        bboxes_pred = bboxes_pred[0, ...]

        labels = torch.arange(self.num_classes, device=scores.device).unsqueeze(0).repeat(cfg.model.instance_head.num_masks, 1).flatten(0, 1)
        scores, topk_indices = scores.flatten(0, 1).topk(cfg.model.instance_head.num_masks, sorted=False)
        labels = labels[topk_indices]

        topk_indices = topk_indices // self.num_classes
        masks_pred = masks_pred[topk_indices]
        iou_scores = iou_scores[topk_indices]
        bboxes_pred = bboxes_pred[topk_indices]



        # masks_pred = pred['pred_masks'].sigmoid()
        # iou_scores = pred['pred_scores'].sigmoid()
        # bboxes_pred = pred['pred_bboxes']
        
        # scores = pred['pred_logits'].sigmoid()
        # scores = scores[0, :, 0]
        # labels = torch.zeros(len(scores), dtype=torch.int64)
        
        # masks_pred = masks_pred[0, ...]
        # scores = scores[0, ...]
        # iou_scores = iou_scores[0, ...].flatten(0, 1)
        # bboxes_pred = bboxes_pred[0, ...]




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
        # print(f"num_preds after 1st cls_thr: {len(scores)}")


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
        # print(scores)
        # print(f"num_preds after nms: {len(scores)}")


        # # ========== CLS Score ==========
        # # score filtering.
        # keep = scores > self.score_threshold
        # masks_pred = masks_pred[keep]
        # scores = scores[keep]
        # labels = labels[keep]
        # print(f"num_preds after 1st cls_thr: {len(scores)}")


        masks_pred = masks_pred > self.mask_threshold
        # ================================================


        results = dict()
        results["img_id"] = pred["img_id"]
        results["ori_shape"] = pred["ori_shape"]
        results["pred_instances"] = {
            "masks": masks_pred,
            "labels": labels,
            "scores": scores,
            "mask_scores": scores,
            "bboxes": bboxes_pred,
        }

        data_samples = [results]
        self.coco_metric.process({}, data_samples)
        

    # def forward(self, dataloader):
    #     super().forward(dataloader)
    #     print("UserWarning: Running Mask2Former evaluation scheme!\n")

    #     for step, batch in enumerate(dataloader):
    #         if batch is None:
    #             continue
            
    #         # prepare targets
    #         images = []
    #         targets = []

    #         target = batch[0]
    #         ignore = ["img_id", "img_path", "ori_shape", "file_name", "coco_id"]
    #         target = {k: v.to(cfg.device) if k not in ignore else v 
    #                 for k, v in target.items()}
    #         images.append(target["image"])
    #         targets.append(target)

    #         image = nested_tensor_from_tensor_list(images)


    #         # ============= PREDICTION ==============
    #         # predict.
    #         pred = self.inference_single(image.tensors)

    #         scores = pred['pred_logits'].softmax(-1)
    #         masks_pred = pred['pred_masks']
            
    #         masks_pred = masks_pred[0, ...]
    #         scores = scores[0, :, :-1]

    #         labels = torch.arange(self.num_classes, device=scores.device).unsqueeze(0).repeat(cfg.model.instance_head.num_masks, 1).flatten(0, 1)
    #         scores, topk_indices = scores.flatten(0, 1).topk(100, sorted=False)
    #         labels = labels[topk_indices]

    #         topk_indices = topk_indices // cfg.model.num_classes
    #         masks_pred = masks_pred[topk_indices]

    #         # maskness scores.
    #         seg_masks = (masks_pred > 0).float()
    #         maskness_scores = (masks_pred.sigmoid().flatten(1) * seg_masks.flatten(1)).sum(1) / (seg_masks.flatten(1).sum(1) + 1e-6)
    #         scores = scores * maskness_scores

    #         # ========== CLS Score ==========
    #         # score filtering.
    #         keep = scores > self.score_threshold
    #         masks_pred = masks_pred[keep]
    #         scores = scores[keep]
    #         labels = labels[keep]
    #         # print(f"num_preds after 1st cls_thr: {len(scores)}")

    #         print(labels)


    #         # ========== NMS ==========
    #         # pre_nms sort.
    #         # sort_inds = torch.argsort(scores, descending=True)
    #         # masks_pred = masks_pred[sort_inds, :, :]
    #         # scores = scores[sort_inds]
    #         # labels = labels[sort_inds]

    #         # # nms.
    #         # # seg_masks = masks_pred.sigmoid() > 0.1
    #         # # sum_masks = seg_masks.sum((1, 2)).float()
    #         # sum_masks = (masks_pred.sigmoid() > self.mask_threshold).sum((1, 2)).float()
    #         # # seg_masks = (masks_pred.sigmoid() > 0.5).float()
    #         # print(sum_masks)
            
    #         # keep = mask_nms(labels, seg_masks, sum_masks, scores, nms_thr=self.nms_threshold)
    #         # keep = keep.bool()
    #         # seg_masks = seg_masks[keep, :, :]
    #         # scores = scores[keep]
    #         # labels = labels[keep]
    #         # print(scores)
    #         # print(f"num_preds after nms: {len(scores)}")


    #         masks_pred = (masks_pred > 0).float()
    #         # ================================================


    #         results = dict()
    #         results["img_id"] = target["img_id"]
    #         results["ori_shape"] = target["ori_shape"]
    #         results["pred_instances"] = {
    #             "masks": masks_pred,
    #             "labels": labels,
    #             "scores": scores,
    #             "mask_scores": scores,
    #             "bboxes": torch.zeros(len(scores), 4),
    #         }

    #         data_samples = [results]
    #         self.coco_metric.process({}, data_samples)


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

