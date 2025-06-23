import logging
import numpy as np
import torch

from models.seg.loss import box_cxcywh_to_xyxy
from utils.visualise import visualize_grid_v2

from .base_visualizer import BaseVisualizer
from utils.registry import CALLBACKS


@CALLBACKS.register(name="InstanceVisualizer")
class InstanceVisualizer(BaseVisualizer):
    """
    Visualizes instance predictions for one or more instance types from model outputs.
    Pass a list to inst_types to visualize multiple types in one callback.
    """
    PRED_LOGITS_KEY = 'pred_logits'
    PRED_BBOXES_KEY = 'pred_bboxes'
    PRED_SCORES_KEY = 'pred_scores'

    def __init__(self, inst_types, ncols, show_bboxes=False, **kwargs):
        super().__init__(**kwargs)
        self.inst_types = inst_types if isinstance(inst_types, list) else [inst_types]
        self.ncols = ncols
        self.show_bboxes = show_bboxes

    def _plot_preds(self, output: dict, save_path: str):
        """
        Plot predictions for all specified instance types present in output.
        """
        if not isinstance(output, dict):
            raise ValueError(f"Expected output to be a dict, got {type(output)}")
        for inst_type in self.inst_types:
            self._plot_single_type(output, save_path, inst_type)

    def _plot_single_type(self, output: dict, save_path: str, inst_type: str):
        """
        Plot predictions for a single instance type.
        """
        key = f'pred_{inst_type}_masks'
        if self.PRED_LOGITS_KEY not in output or self.PRED_BBOXES_KEY not in output:
            logging.warning(f"Missing required keys in output for {inst_type}. Skipping.")
            return
        if key not in output:
            logging.info(f"Mask key {key} not found in output. Skipping {inst_type}.")
            return
        masks, bboxes, titles = self._extract_and_sort(output, key)
        self._plot_grid(masks, bboxes, titles, save_path, inst_type)

    def _extract_and_sort(self, output: dict, key: str):
        """
        Extract and sort masks, bboxes, and titles by score.
        """
        masks = output[key].sigmoid().cpu().detach().numpy()
        probs = output[self.PRED_LOGITS_KEY].softmax(-1)
        scores = probs[0, :, 0].cpu().detach().numpy()

        if output.get(self.PRED_SCORES_KEY) is not None:
            iou_scores = output[self.PRED_SCORES_KEY].sigmoid()
            scores = iou_scores[0, :, 0].cpu().detach().numpy()
        else:
            iou_scores = np.zeros_like(scores)
        
        h, w = output[key].shape[-2:]
        bboxes = box_cxcywh_to_xyxy(output[self.PRED_BBOXES_KEY])
        bboxes = bboxes.cpu().detach() * torch.tensor([w, h, w, h], dtype=torch.float32)
        bboxes = bboxes.numpy()
        titles = [f"conf: {score:.2f}" for score in scores]
        idx = np.argsort(-scores)
        
        masks = masks[0][idx]
        bboxes = bboxes[0][idx]
        titles = [titles[i] for i in idx]

        return masks, bboxes, titles

    def _plot_grid(self, masks, bboxes, titles, save_path, inst_type):
        """
        Plot grid of masks and bboxes.
        """
        nrows = ncols = self.ncols
        num_masks = masks.shape[0]
        num_grids = (num_masks + nrows * ncols - 1) // (nrows * ncols)
        
        for grid_idx in range(num_grids):
            start_idx = grid_idx * nrows * ncols
            end_idx = min(start_idx + nrows * ncols, num_masks)
            grid_masks = masks[start_idx:end_idx]
            grid_bboxes = bboxes[start_idx:end_idx] if self.show_bboxes else None
            grid_titles = titles[start_idx:end_idx]
            
            visualize_grid_v2(
                masks=grid_masks,
                bboxes=grid_bboxes,
                titles=grid_titles,
                ncols=self.ncols,
                nrows=nrows,
                path=f'{save_path}/{inst_type}/batch_{grid_idx}/pred_masks.jpg'
            )



