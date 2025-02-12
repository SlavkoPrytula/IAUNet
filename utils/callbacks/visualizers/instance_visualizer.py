from os import makedirs
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn.functional as F

import sys
sys.path.append("./")

from models.seg.loss import box_cxcywh_to_xyxy
from utils.visualise import visualize, visualize_grid, visualize_grid_v2

from .base_visualizer import BaseVisualizer
from utils.registry import CALLBACKS


@CALLBACKS.register(name="InstanceVisualizer")
class InstanceVisualizer(BaseVisualizer):
    def __init__(self, inst_type, ncols, show_bboxes=False, **kwargs):
        super().__init__(**kwargs)
        self.inst_type = inst_type
        self.ncols = ncols
        self.show_bboxes = show_bboxes
        
    def plot(self, cfg, output, save_path):
        self.plot_preds(cfg, output, save_path)
        self.plot_aux_preds(cfg, output, save_path)

    def _plot_preds(self, output, save_path):
        """
        Pred Mask Visuals
        """
        assert 'pred_logits' in output, f"pred_logits not in output"
        assert 'pred_bboxes' in output, f"pred_bboxes not in output"
        
        if not f'pred_{self.inst_type}_masks' in output:
            return 
        
        # -----------
        # Pred Masks + BBoxes.
        vis_preds_inst = output[f'pred_{self.inst_type}_masks'].sigmoid().cpu().detach().numpy()
        probs = output['pred_logits'].softmax(-1)
        scores = probs[0, :, 0].cpu().detach().numpy()

        if 'pred_scores' in output:
            iou_scores = output['pred_scores'].sigmoid()
            iou_scores = iou_scores[0, :, 0].cpu().detach().numpy()
        else:
            iou_scores = np.zeros_like(scores)

        h, w = output[f'pred_{self.inst_type}_masks'].shape[-2:]
        bboxes = box_cxcywh_to_xyxy(output["pred_bboxes"])
        bboxes = bboxes.cpu().detach()
        bboxes = bboxes * torch.tensor([w, h, w, h], dtype=torch.float32)
        bboxes = bboxes.numpy()

        titles = [
            f"conf: {score:.2f}, iou: {iou_score:.2f}"
            for score, iou_score in zip(scores, iou_scores)
        ]

        # -----------
        # sort.
        idx = np.argsort(-scores)
        vis_preds_inst = vis_preds_inst[0][idx]
        bboxes = bboxes[0][idx]
        titles = [titles[i] for i in idx]
        
        # -----------
        # grid plot.
        nrows = ncols = self.ncols
        num_masks = vis_preds_inst.shape[0]
        num_grids = (num_masks + nrows * ncols - 1) // (nrows * ncols)

        for grid_idx in range(num_grids):
            start_idx = grid_idx * nrows * ncols
            end_idx = min(start_idx + nrows * ncols, num_masks)

            grid_masks = vis_preds_inst[start_idx:end_idx]
            grid_bboxes = bboxes[start_idx:end_idx] if self.show_bboxes else None
            grid_titles = titles[start_idx:end_idx]

            visualize_grid_v2(
                masks=grid_masks, 
                bboxes=grid_bboxes,
                titles=grid_titles,
                ncols=self.ncols, 
                nrows=nrows,
                path=f'{save_path}/{self.inst_type}/batch_{grid_idx}/pred_masks.jpg'
            )

        
    def plot_preds(self, cfg, output, save_path):
        self._plot_preds(output, save_path=save_path)

    def plot_aux_preds(self, cfg, output, save_path):
        """
        Aux Mask Visuals
        """
        # Aux Pred Masks.
        if "aux_outputs" in output:
            for i, aux_outputs in enumerate(output['aux_outputs']):
                self._plot_preds(aux_outputs, save_path=f"{save_path}/aux_outputs/layer_{i}")



