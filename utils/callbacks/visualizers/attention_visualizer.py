import sys
sys.path.append("./")

import matplotlib.pyplot as plt
import numpy as np

from utils.visualise import visualize, visualize_grid_v2
from .base_visualizer import BaseVisualizer
from utils.registry import CALLBACKS


@CALLBACKS.register(name="AttentionVisualizer")
class AttentionVisualizer(BaseVisualizer):
    def __init__(self, inst_type, ncols, **kwargs):
        super().__init__(**kwargs)
        self.inst_type = inst_type
        self.ncols = ncols
        
    def plot(self, cfg, output, save_path):
        self.plot_preds(cfg, output, save_path)
        self.plot_aux_preds(cfg, output, save_path)


    def _plot_sa_attn(self, output, save_path, type='query_sa_attn'):
        if not type in output[f'attn']:
            return 
        if output[f'attn'][type] is None:
            return 
        
        attn = output[f'attn'][type].cpu().detach().numpy() # b, n, n
        attn = attn[0]

        visualize(
            figsize=[15, 15],
            attn=attn,
            show_title=False,
            path=f'{save_path}/{self.inst_type}/attn/{type}.jpg'
        )


    def _plot_ca_attn(self, output, save_path, type='inst_pixel_attn'):
        # function to visuzalise cross-attention between (b, c, h, w) features and (b, n, d) queries
        # intput attention maps come in shape (b, n, h, w) from the model
        if not type in output[f'attn']:
            return 
        if output[f'attn'][type] is None:
            return 
        
        attn = output[f'attn'][type].cpu().detach().numpy() # b, n, h, w
        attn = attn[0]
        # split double queries - hack to be fixed using N from num_masks
        attn1 = attn[:100]
        attn2 = attn[100:]

        visualize_grid_v2(
            figsize=[15, 15],
            masks=attn1,
            ncols=self.ncols,
            path=f'{save_path}/{self.inst_type}/attn/{type}_q1.jpg'
        )

        visualize_grid_v2(
            figsize=[15, 15],
            masks=attn2,
            ncols=self.ncols,
            path=f'{save_path}/{self.inst_type}/attn/{type}_q2.jpg'
        )


    def _plot_preds(self, output, save_path):
        if not 'attn' in output:
            return 

        self._plot_ca_attn(output, save_path, type='inst_pixel_attn')
        self._plot_ca_attn(output, save_path, type='mask_pixel_attn')
        self._plot_sa_attn(output, save_path, type='query_sa_attn')
        
        
    def plot_preds(self, cfg, output, save_path):
        self._plot_preds(output, save_path=save_path)

    def plot_aux_preds(self, cfg, output, save_path):
        # Aux Pred Masks.
        if "aux_outputs" in output:
            for i, aux_outputs in enumerate(output['aux_outputs']):
                self._plot_preds(aux_outputs, save_path=f"{save_path}/aux_outputs/layer_{i}")



