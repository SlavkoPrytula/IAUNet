import torch
import sys
sys.path.append("./")

from evaluation.metrics import IOUMetric

iou_metric = IOUMetric()

gt_masks = torch.ones(512, 512)
pred_masks = torch.ones(512, 512)
data_samples = [{
    "gt_masks": gt_masks, 
    "pred_masks": pred_masks
    }]

iou_metric.process(None, data_samples)
eval_results = iou_metric.compute_metrics(iou_metric.results)

print(f"Mean IoU for empty masks: {eval_results['mean_IoU']}")
