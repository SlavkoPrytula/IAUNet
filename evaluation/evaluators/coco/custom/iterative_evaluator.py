import torch
from os.path import join
from configs import cfg
from tqdm import tqdm

from ..mmdet_dataloader_evaluator import MMDetDataloaderEvaluator

from utils.utils import nested_tensor_from_tensor_list
from utils.opt.mask_nms import mask_nms
from utils.registry import EVALUATORS, DATASETS
from visualizations import save_coco_vis

from utils.common.decorators import timeit_evaluator, memory_evaluator


@timeit_evaluator
@memory_evaluator
@EVALUATORS.register(name="IterativeEvaluator")
class IterativeEvaluator(MMDetDataloaderEvaluator):
    def __init__(self, cfg: cfg, model=None, dataset=None, **kwargs):
        super().__init__(cfg, model, dataset, **kwargs)
        self.max_iters = cfg.model.evaluator.max_iters if cfg.model.evaluator.max_iters is not None else len(dataset)
        # remove gt annotations to use only one gt anns per iteration (gt by id).
        self.reset()

    def forward(self, dataloader):
        pbar = tqdm(enumerate(dataloader), total=self.max_iters, miniters=5)
        for step, batch in pbar:
            if step > self.max_iters - 1:
                break
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

            print(f"=" * 50)
            print()
            print(f'Processing image {step} with img_id {target["img_id"]}, file_name {target["file_name"]}')
            print()
            self.process(pred, target)
            self.evaluate()
            self.visualize(target)
            self.reset()
            print()
            print()


    def visualize(self, target):
        gt_coco = self.gt_coco
        pred_coco = self.pred_coco

        img = target["image"][0].cpu().numpy()
        fname = target["file_name"]
        idx = target["coco_id"]
        H, W = target["ori_shape"]
        out_file = join(self.cfg.visuals_dir, f'{fname}.jpg')

        save_coco_vis(img, gt_coco, pred_coco, idx, shape=[H, W], path=out_file, show_img=True)


    def reset(self):
        # simple mmdet coco api reset.
        self.coco_metric._coco_api = None
        self.coco_metric.cat_ids = None
        self.coco_metric.img_ids = None

    
    def process(self, pred: dict, target: dict):
        scores = pred['pred_logits'].softmax(-1)
        masks_pred = pred['pred_masks'].sigmoid()
        iou_scores = pred['pred_scores'].sigmoid()
        bboxes_pred = pred['pred_bboxes']
        
        masks_pred = masks_pred[0, ...]
        scores = scores[0, :, :-1]
        iou_scores = iou_scores[0, ...].flatten(0, 1)
        bboxes_pred = bboxes_pred[0, ...]

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

        # add gt anns.
        results["instances"] = {
            "masks": target["masks"],
            "bbox_labels": target["labels"],
            "bboxes":  target["bboxes"] if "bboxes" in target else torch.zeros(len(scores), 4)
        }

        data_samples = [results]
        self.coco_metric.process({}, data_samples)
        

    