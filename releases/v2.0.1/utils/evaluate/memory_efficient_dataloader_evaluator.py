import numpy as np
import torch
from torch import nn

# from utils.coco.coco import COCO
from utils.coco.mask2coco import masks2coco
import pycocotools.mask as mask_util
import datetime
# from pycocotools.cocoeval import COCOeval

from .coco_evaluator import Evaluator
from utils.utils import nested_tensor_from_tensor_list

from configs import cfg
import matplotlib.pyplot as plt

from utils.utils import flatten_mask
from utils.opt.mask_nms import mask_nms

from utils.registry import EVALUATORS



def get_coco_template():
    return {
        "info": {
            "year": "2023",
            "version": "1.0",
            "description": "Exported using ChatGPT's adaptation",
            "contributor": "",
            "url": "",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        },
        "licenses": [],
        "categories": [
            {"id": 1, "name": "object", "supercategory": "object"},
        ],
        "images": [],
        "annotations": []
    }

def create_image_info(image_id, file_name, image_size):
    """Create the image info section for a COCO data point."""
    width, height = image_size
    return {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
        "date_captured": datetime.datetime.utcnow().isoformat(' '),
        "license": 1,
        "coco_url": "",
        "flickr_url": ""
    }

def create_annotation_info(annotation_id, image_id, binary_mask, scores=None, category_id=1):
    """Create the annotation info section for a COCO data point."""
    rle = mask_util.encode(np.asfortranarray(binary_mask))
    rle['counts'] = rle['counts'].decode('utf-8')

    area = np.sum(binary_mask)  # count the number of 1s in the binary mask (true pixels)
    bbox = list(mask_util.toBbox(rle))


    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": rle,
        "area": area,
        "bbox": bbox,
        "iscrowd": 0
    }

    if scores is not None:
        annotation_info['score'] = scores
    else:
        annotation_info['score'] = 1

    return annotation_info

def masks_to_coco_json(masks_list, image_paths, scores_list=None):
    coco_output = get_coco_template()

    annotation_id = 1
    for image_id, (image_path, masks) in enumerate(zip(image_paths, masks_list), start=1):
        image_size = masks.shape[1:] if masks.ndim > 2 else masks.shape  # (H, W) assuming masks could be (N, H, W) or (H, W)
        coco_output["images"].append(create_image_info(image_id, image_path, image_size))

        if masks.ndim > 2:  # Check if masks contain N dimension
            for mask_num, mask in enumerate(masks):
                score = scores_list[image_id-1][mask_num] if scores_list is not None else None
                annotation_info = create_annotation_info(annotation_id, image_id, mask, score)
                coco_output["annotations"].append(annotation_info)
                annotation_id += 1
        else:  # There's only one mask
            score = scores_list[image_id-1] if scores_list is not None else None
            annotation_info = create_annotation_info(annotation_id, image_id, masks, score)
            coco_output["annotations"].append(annotation_info)
            annotation_id += 1

    return coco_output


# TODO: merge base and nms evaluators
@EVALUATORS.register(name="MemoryEfficientDataloaderEvaluator")
class MemoryEfficientDataloaderEvaluator(Evaluator):
    # coco_eval
    def __init__(self, cfg: cfg):
        super(MemoryEfficientDataloaderEvaluator, self).__init__(cfg)
        self.gt_coco = get_coco_template()
        self.pred_coco = get_coco_template()
        self.annotation_id = 1 

    def forward(self, model, dataloader):
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
            output = self.inference_single(model, image.tensors)

            pred = output
            scores = pred['pred_logits'].sigmoid()
            scores = scores[0, :, 0]

            masks_pred = pred['pred_masks'].sigmoid()
            masks_pred = masks_pred[0, ...]
            
            # maskness scores.
            maskness_scores = []
            for p in masks_pred:
                maskness_score = torch.mean(p[p.gt(self.mask_threshold)])
                maskness_score = torch.nan_to_num(maskness_score, nan=0.0)
                maskness_score = maskness_score.cpu()
                maskness_scores.append(maskness_score)

            maskness_scores = torch.tensor(maskness_scores).to(cfg.device)
            scores = maskness_scores

            # masks_pred = masks_pred[scores > 0.4]
            # scores = scores[scores > 0.4]

            scores = scores.detach().cpu().numpy()
            masks_pred = masks_pred.detach().cpu().numpy()
            masks_pred = (masks_pred > self.mask_threshold).astype(np.uint8)

            masks = target['masks']
            masks = masks.detach().cpu().numpy()

            # store data.
            for idx, (gt_mask, pred_mask, score) in enumerate(zip(masks, masks_pred, scores)):
                gt_mask = gt_mask.astype(np.uint8)
                pred_mask = pred_mask.astype(np.uint8)

                image_info = create_image_info(image_id=step, file_name=f"image_{step}.jpg", image_size=pred_mask.shape[-2:])
                self.gt_coco["images"].append(image_info)
                self.pred_coco["images"].append(image_info)

                # GT annotations
                gt_annotation_info = create_annotation_info(annotation_id=self.annotation_id, image_id=step, binary_mask=gt_mask)
                self.gt_coco["annotations"].append(gt_annotation_info)

                # Predicted annotations
                pred_annotation_info = create_annotation_info(annotation_id=self.annotation_id, image_id=step, binary_mask=pred_mask, scores=score)
                self.pred_coco["annotations"].append(pred_annotation_info)

                self.annotation_id += 1

        # masks2coco
        # self.gt_coco = masks2coco(gt_masks)
        # self.pred_coco = masks2coco(pred_masks, scores=pred_scores)

