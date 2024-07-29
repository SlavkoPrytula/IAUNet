from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from tqdm import tqdm
import json
import tempfile

import sys
sys.path.append(".")
# sys.path.append("..")

from evaluation.mmdet.metrics import CocoMetric


def process_coco(json_file_path, default_score=0.9):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    for annotation in data['annotations']:
        annotation['score'] = default_score

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as tmpfile:
        json.dump(data, tmpfile)
        temp_file_name = tmpfile.name

    return temp_file_name


def main():
    dataset = DATASETS.get(cfg.dataset.type)(cfg, dataset_type="eval")

    coco_metric = CocoMetric(
        ann_file=dataset.ann_file, 
        metric='segm', 
        classwise=True,
        # outfile_prefix="./test/temp"
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
    for idx in tqdm(range(4)):
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


    # ========================================
    coco_gt = COCO(dataset.ann_file)
    coco_pred = COCO(process_coco(dataset.ann_file))
    
    coco_eval = COCOeval(coco_gt, coco_pred, 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    from dataset.datasets import HuBMAP, LiveCell30Images, BrightfieldCOCO
    from configs.datasets import HuBMAP, LiveCell30Images, BrightfieldCOCO_v2
    from utils.registry import DATASETS_CFG, DATASETS
    from configs import cfg

    cfg.dataset = DATASETS_CFG.get("brightfield_coco_v2.0")

    main()
