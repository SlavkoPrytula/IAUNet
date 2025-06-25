import sys
sys.path.append("./")

import matplotlib.pyplot as plt
import numpy as np

from visualizations.visualise import visualize
from .base_visualizer import BaseVisualizer
from utils.registry import CALLBACKS


@CALLBACKS.register(name="FeaturesVisualizer")
class FeaturesVisualizer(BaseVisualizer):
    def __init__(self, inst_type, **kwargs):
        super().__init__(**kwargs)
        self.inst_type = inst_type
        
    def plot(self, cfg, output, save_path):
        self.plot_preds(cfg, output, save_path)
        self.plot_aux_preds(cfg, output, save_path)


    def _plot_feats(self, output, save_path, type='mask_feats'):
        if not type in output[f'pred_{self.inst_type}_feats']:
            return 
        
        vis_pred_feats = output[f'pred_{self.inst_type}_feats'][type].cpu().detach().numpy()
        vis_pred_feats = vis_pred_feats[0]

        visualize(
            figsize=[15, 5],
            mask_feats1=vis_pred_feats[0], 
            mask_feats2=vis_pred_feats[1], 
            mask_feats3=vis_pred_feats[2], 
            mask_feats4=vis_pred_feats[3], 
            show_title=False,
            path=f'{save_path}/{self.inst_type}/{type}.jpg'
        )

    def _plot_attention(self, output, save_path, type='attn_mask'):
        if 'attn_mask' not in output[f'pred_{self.inst_type}_feats']:
            return

        attn_mask = output[f'pred_{self.inst_type}_feats']['attn_mask'].cpu().detach()
        masks = output[f'pred_{self.inst_type}_masks'].sigmoid().cpu().detach().numpy()
        probs = output['pred_logits'].softmax(-1)
        scores = probs[0, :, 0].cpu().detach().numpy()

        B, nhead, N, H, W = attn_mask.shape
        nhead = 1

        # -----------
        # sort.
        idx = np.argsort(-scores)
        masks = masks[0][idx]
        attn_mask = attn_mask[0][:, idx]

        nrows = 15
        fig, axs = plt.subplots(nrows, nhead+1, figsize=((nhead+1)*2, 30))
        
        for i in range(nrows):
            for j in range(nhead+1):
                ax = axs[i, j]
                ax.axis('off')

                if j == 0:
                    ax.imshow(masks[i, :, :], cmap='viridis')
                    ax.set_title(f'pred {i}', fontsize=10)
                else:
                    ax.imshow(attn_mask[j-1, i, :, :], cmap='jet')
                    ax.set_title(f'head {j-1}', fontsize=10)

        fig.tight_layout(pad=0.5)
        plt.savefig(f"{save_path}/{self.inst_type}/{type}.jpg")
        plt.close(fig)
        

    def _plot_preds(self, output, save_path):
        if not f'pred_{self.inst_type}_feats' in output:
            return 

        self._plot_feats(output, save_path, type='mask_feats')
        self._plot_feats(output, save_path, type='inst_feats')
        self._plot_attention(output, save_path, type='attn_mask')
        
        
    def plot_preds(self, cfg, output, save_path):
        self._plot_preds(output, save_path=save_path)

    def plot_aux_preds(self, cfg, output, save_path):
        # Aux Pred Masks.
        if "aux_outputs" in output:
            for i, aux_outputs in enumerate(output['aux_outputs']):
                self._plot_preds(aux_outputs, save_path=f"{save_path}/aux_outputs/layer_{i}")



