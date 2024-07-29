import torch
from tqdm import tqdm
import numpy as np

import sys
sys.path.append(".")

from evaluation.mmdet.metrics import CocoMetric
from dataset.dataloaders import build_loader
from utils.utils import nested_tensor_from_tensor_list
from utils.augmentations import train_transforms, valid_transforms


def main():
    dataset = DATASETS.get(cfg.dataset.type)(cfg, dataset_type="valid")
    occ_dataset = DATASETS.get(cfg.dataset.type)(cfg, dataset_type="occ")
    dataloader = build_loader(dataset, 
                              batch_size=1)

    coco_metric = CocoMetric(
        ann_file=occ_dataset.ann_file, 
        metric='segm', 
        classwise=True,
        coco_api='COCOeval_nofp'
        )

    categories = coco_metric._coco_api.loadCats(coco_metric._coco_api.getCatIds())
    class_names = [category['name'] for category in categories]
    coco_metric.dataset_meta = dict(classes=class_names)


    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
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

        preds = {
            "pred_masks": target["masks"],
            "pred_labels": target["labels"]
        }
 
        results = {
            "pred_instances": {},
            "img_id": None,
            "ori_shape": None,
        }   

        print(preds["pred_labels"])

        results["img_id"] = target["img_id"]
        results["ori_shape"] = target["ori_shape"]
        results["pred_instances"] = {
            "masks": preds["pred_masks"],
            "labels": preds["pred_labels"],
            "scores": torch.zeros(len(preds["pred_masks"])),
            # "mask_scores": torch.ones(len(preds["pred_masks"])),
            "bboxes": torch.zeros(len(preds["pred_masks"]), 4),
        }

        data_samples = [results]
        coco_metric.process({}, data_samples)

    # Compute metrics
    eval_results = coco_metric.evaluate()
    print(eval_results)


if __name__ == "__main__":
    from dataset.datasets import HuBMAP, LiveCell30Images, YeastNet, Rectangle
    from configs.datasets import HuBMAP, LiveCell30Images, YeastNet, Rectangle
    from configs import cfg
    from utils.registry import DATASETS_CFG, DATASETS

    cfg.dataset = DATASETS_CFG.get("brightfield_coco_v2.0")

    main()
