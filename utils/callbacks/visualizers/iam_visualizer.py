import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from os.path import join

import sys
sys.path.append("./")

from visualizations.visualise import visualize, visualize_grid, visualize_grid_v2
from .base_visualizer import BaseVisualizer
from utils.registry import CALLBACKS


@CALLBACKS.register(name="IAMVisualizer")
class IAMVisualizer(BaseVisualizer):
    def __init__(self, inst_type, ncols, nrows, **kwargs):
        super().__init__(**kwargs)
        self.inst_type = inst_type
        self.ncols = ncols
        self.nrows = nrows
        
    def plot(self, cfg, output, save_path):
        self.plot_iam_preds(cfg, output, save_path)
        self.plot_aux_iam_preds(cfg, output, save_path)
        self.plot_iam_heads(cfg, output, save_path)


    def _plot_iam(self, iams, titles, save_path, batch_idx, mode='logits'):
        batch_folder = join(save_path, self.inst_type, f'batch_{batch_idx}')
        visualize_grid_v2(
            masks=iams, 
            titles=titles,
            ncols=self.ncols, 
            path=join(batch_folder, f'[pred_iam]_{mode}.jpg'),
            cmap='jet', # plasma
        )


    def plot_logits_iam(self, iams, titles, save_path, batch_idx):
        # -----------
        # IAM Logits. 
        vis_preds_iams = iams.clone().cpu().detach().numpy()
        self._plot_iam(vis_preds_iams, titles, save_path, batch_idx, mode='logits')


    def plot_softmax_iam(self, iams, titles, save_path, batch_idx):
        # -----------
        # IAM Softmax.  
        N, H, W = iams.shape
        _iam = iams.clone()
        _iam = F.softmax(_iam.view(N, -1), dim=-1)
        _iam = _iam.view(N, H, W)
        vis_preds_iams = _iam.cpu().detach().numpy()
        self._plot_iam(vis_preds_iams, titles, save_path, batch_idx, mode='softmax')
    

    def plot_sigmoid_iam(self, iams, titles, save_path, batch_idx):
        # -----------
        # IAM Sigmoid.
        vis_preds_iams = iams.clone().sigmoid().cpu().detach().numpy()
        self._plot_iam(vis_preds_iams, titles, save_path, batch_idx, mode='sigmoid')


    def _plot_preds(self, output, save_path):
        """
        Pred IAMs Visuals
        """
        if 'pred_iams' not in output or output['pred_iams'] is None:
            return
        if not f'{self.inst_type}_iams' in output['pred_iams']:
            return 
        
        iams = output['pred_iams'][f'{self.inst_type}_iams']

        probs = output['pred_logits'].softmax(-1)
        scores = probs[0, ...].cpu().detach().numpy()
        scores = np.round(scores, 3)
        titles = [', '.join([f"({class_idx}, {score:.2f})" for class_idx, score in 
                             zip(range(scores.shape[1]), score)]) for score in scores]

        # -----------
        # sort.
        idx = np.argsort(-scores[:, 0])
        iams = iams[0][idx]
        titles = [titles[i] for i in idx]

        # -----------
        # grid plot.
        nrows = ncols = self.ncols
        num_masks = iams.shape[0]
        num_grids = (num_masks + nrows * ncols - 1) // (nrows * ncols)

        for grid_idx in range(num_grids):
            start_idx = grid_idx * nrows * ncols
            end_idx = min(start_idx + nrows * ncols, num_masks)

            grid_masks = iams[start_idx:end_idx]
            grid_titles = titles[start_idx:end_idx]

            self.plot_logits_iam(grid_masks, grid_titles, save_path, batch_idx=grid_idx)
            self.plot_softmax_iam(grid_masks, grid_titles, save_path, batch_idx=grid_idx)
            self.plot_sigmoid_iam(grid_masks, grid_titles, save_path, batch_idx=grid_idx)
        
        
    def plot_iam_preds(self, cfg, output, save_path):
        self._plot_preds(output, save_path=save_path)


    def plot_aux_iam_preds(self, cfg, output, save_path):
        """
        Aux Mask Visuals
        """
        # Aux Pred Masks.
        if "aux_outputs" in output:
            for i, aux_outputs in enumerate(output['aux_outputs']):
                self._plot_preds(aux_outputs, save_path=f"{save_path}/aux_outputs/layer_{i}")


    # ==============
    # Multiple Heads.
    # ==============
    def _plot_iam_heads(self, cfg, masks, iams, save_path, mode=''): 
        N, H, W = iams.shape
        # groups = cfg.model.decoder.instance_head.num_groups
        groups = 1

        fig, axs = plt.subplots(self.nrows, groups+1, figsize=((groups+1)*2, 30))
        for i in range(self.nrows):
            for j in range(groups+1):
                ax = axs[i, j]
                ax.axis('off')

                if j == 0:
                    ax.imshow(masks[i, :, :], cmap='viridis')
                    ax.set_title(f'pred {i}', fontsize=10)
                else:
                    channel_idx = N // groups * (j-1) + i
                    ax.imshow(iams[channel_idx, :, :], cmap='jet')
                    ax.set_title(f'head {j-1}', fontsize=10)
        
        fig.tight_layout(pad=0.5)
        plt.savefig(f'{save_path}/{self.inst_type}/[pred_iam]_{mode}_grouped.jpg')
        plt.close()


    def plot_logits_iam_heads(self, cfg, masks, iams, save_path):
        # -----------
        # IAM Logits [Grouped]. 
        iams = iams.clone().cpu().detach().numpy()
        self._plot_iam_heads(cfg, masks, iams, save_path, mode='logits')


    def plot_softmax_iam_heads(self, cfg, masks, iams, save_path):
        # -----------
        # IAM Softmax [Grouped].  
        N, H, W = iams.shape
        _iam = iams.clone()
        _iam = F.softmax(_iam.view(N, -1), dim=-1)
        _iam = _iam.view(N, H, W)
        iams = _iam.cpu().detach().numpy()
        self._plot_iam_heads(cfg, masks, iams, save_path, mode='softmax')


    def plot_sigmoid_iam_heads(self, cfg, masks, iams, save_path):
        # -----------
        # IAM Sigmoid [Grouped]. 
        iams = iams.clone().sigmoid().cpu().detach().numpy()
        self._plot_iam_heads(cfg, masks, iams, save_path, mode='sigmoid')


    def plot_iam_heads(self, cfg, output, save_path):
        if output['pred_iams'] is None:
            return
        if not f'{self.inst_type}_iams' in output['pred_iams']:
            return 
        
        masks = output[f'pred_{self.inst_type}_masks'].sigmoid().cpu().detach().numpy()
        iams = output['pred_iams'][f'{self.inst_type}_iams']
        probs = output['pred_logits'].softmax(-1)
        scores = probs[0, ...].cpu().detach().numpy()

        # -----------
        # sort.
        idx = np.argsort(-scores[:, 0])
        masks = masks[0][idx]

        B, heads_n, H, W = iams.shape
        # heads = cfg.model.decoder.instance_head.num_groups
        heads = 1
        N = heads_n // heads
        iams = iams.view(B, heads, N, H, W)
        iams = iams[0][:, idx]
        iams = iams.view(-1, H, W)
        
        self.plot_logits_iam_heads(cfg, masks, iams, save_path)
        self.plot_softmax_iam_heads(cfg, masks, iams, save_path)
        self.plot_sigmoid_iam_heads(cfg, masks, iams, save_path)
