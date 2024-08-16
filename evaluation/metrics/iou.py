import numpy as np
from torch import Tensor
from typing import Optional, Union, List, Any, Sequence

from ..mmdet.metrics.base_metric import BaseMetric


class IOUMetric(BaseMetric):
    # default_prefix: Optional[str] = "IoU"

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        """Process a batch of data samples and predictions."""
        for data in data_samples:
            gt_masks = data["gt_masks"]
            pred_masks = data["pred_masks"]

            pred_masks = pred_masks.unsqueeze(0)
            gt_masks = gt_masks.unsqueeze(0)
            pred_masks = pred_masks.flatten(1)  # (1, h*w)
            gt_masks = gt_masks.flatten(1)      # (1, h*w)

            iou_score = self.compute_iou(pred_masks, gt_masks)
            self.results.append({"iou_score": iou_score})

    def compute_metrics(self, results: list) -> dict:
        """Compute the average IoU from processed results."""
        print(f'Evaluating overlap iou...')
        all_iou_scores = []
        for result in results:
            all_iou_scores.extend(result["iou_score"])

        mean_iou = np.mean(all_iou_scores)
        return {"mean_IoU": mean_iou}

    @staticmethod
    def compute_iou(pred_masks, gt_masks):
        intersection = (pred_masks * gt_masks).sum(-1)
        union = pred_masks.sum(-1) + gt_masks.sum(-1) - intersection
        score = intersection / (union + 1e-6)

        return score

