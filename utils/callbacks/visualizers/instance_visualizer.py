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
        if not f'pred_{self.inst_type}_masks' in output:
            return 
        
        # -----------
        # Pred Masks + BBoxes.
        vis_preds_inst = output[f'pred_{self.inst_type}_masks'].sigmoid().cpu().detach().numpy()
        probs = output['pred_logits'].softmax(-1)
        scores = probs[0, :, 0].cpu().detach().numpy()
        scores = np.round(scores, 2)

        iou_scores = output['pred_scores'].sigmoid()
        iou_scores = iou_scores[0, :, 0].cpu().detach().numpy()
        iou_scores = np.round(iou_scores, 2)

        h, w = output[f'pred_{self.inst_type}_masks'].shape[-2:]
        bboxes = box_cxcywh_to_xyxy(output["pred_bboxes"])
        bboxes = bboxes.cpu().detach()
        bboxes = bboxes * torch.tensor([w, h, w, h], dtype=torch.float32)
        bboxes = bboxes.numpy()

        titles = [
            f"conf: {score:.2f}, iou: {iou_score:.2f}"
            for score, iou_score in zip(scores, iou_scores)
        ]
        
        visualize_grid_v2(
            masks=vis_preds_inst[0, ...], 
            bboxes=bboxes[0, ...] if self.show_bboxes else None,
            titles=titles,
            ncols=self.ncols, 
            path=f'{save_path}/{self.inst_type}/pred_masks.jpg'
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
                self._plot_preds(aux_outputs, save_path=f"{save_path}/aux_outputs")



