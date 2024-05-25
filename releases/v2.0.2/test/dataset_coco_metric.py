from pycocotools.coco import COCO
from pycocotools.mask import encode as mask_encode
import numpy as np
import torch
from tqdm import tqdm

import sys
sys.path.append(".")
# sys.path.append("..")

from evaluation.mmdet.metrics import CocoMetric


def main():
    dataset = DATASETS.get(cfg.dataset.name)(cfg, dataset_type="eval")

    coco_metric = CocoMetric(
        ann_file=dataset.ann_file, 
        metric='segm', 
        classwise=True,
        outfile_prefix="./test/temp"
        )

    categories = coco_metric._coco_api.loadCats(coco_metric._coco_api.getCatIds())
    class_names = [category['name'] for category in categories]
    coco_metric.dataset_meta = dict(classes=class_names)

    # Evaluation loop
    for idx in tqdm(range(len(dataset))):
        img_data = dataset[idx]

        preds = {
            "pred_masks": img_data["masks"],
            "pred_labels": img_data["labels"]
        }
 
        results = {
            "pred_instances": {},
            "img_id": None,
            "ori_shape": None,
        }

        results["img_id"] = img_data["img_id"]
        results["ori_shape"] = img_data["ori_shape"]
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


     # Evaluation loop
    for idx in tqdm(range(10)):
        img_data = dataset[idx]

        preds = {
            "pred_masks": img_data["masks"],
            "pred_labels": img_data["labels"]
        }
 
        results = {
            "pred_instances": {},
            "img_id": None,
            "ori_shape": None,
        }

        results["img_id"] = img_data["img_id"]
        results["ori_shape"] = img_data["ori_shape"]
        results["pred_instances"] = {
            "masks": preds["pred_masks"],
            "labels": preds["pred_labels"],  #torch.zeros(len(preds["pred_masks"]), dtype=torch.int64),
            "scores": torch.zeros(len(preds["pred_masks"])),
            # "mask_scores": torch.ones(len(preds["pred_masks"])),
            "bboxes": torch.zeros(len(preds["pred_masks"]), 4),
        }

        data_samples = [results]
        coco_metric.process({}, data_samples)

    # Compute metrics
    eval_results = coco_metric.compute_metrics(coco_metric.results)
    print(eval_results)


if __name__ == "__main__":
    from dataset.datasets import HuBMAP, LiveCell30Images
    from configs.datasets import HuBMAP, LiveCell30Images
    from configs import cfg

    from utils.registry import DATASETS_CFG, DATASETS
    cfg.dataset = DATASETS_CFG.get("HuBMAP")

    main()
