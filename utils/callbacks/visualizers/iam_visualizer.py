import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

import sys
sys.path.append("./")

from utils.visualise import visualize, visualize_grid, visualize_grid_v2

from configs import cfg
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
        self.plot_preds(cfg, output, save_path)
        self.plot_aux_preds(cfg, output, save_path)
        self.plot_iam_heads(cfg, output, save_path)


    def plot_logits_iam(self, iams, titles, save_path):
        # -----------
        # IAM Logits. 
        vis_preds_iams = iams.clone().cpu().detach().numpy()
        
        visualize_grid_v2(
            masks=vis_preds_iams[0, ...], 
            titles=titles,
            ncols=self.ncols, 
            path=f'{save_path}/{self.inst_type}/[pred_iam]_logits.jpg',
            cmap='jet',
        )


    def plot_softmax_iam(self, iams, titles, save_path):
        # -----------
        # IAM Softmax.  
        B, N, H, W = iams.shape
        _iam = iams.clone()
        _iam = F.softmax(_iam.view(B, N, -1), dim=-1)
        _iam = _iam.view(B, N, H, W)
        vis_preds_iams = _iam.cpu().detach().numpy()
        
        visualize_grid_v2(
            masks=vis_preds_iams[0, ...], 
            titles=titles,
            ncols=self.ncols, 
            path=f'{save_path}/{self.inst_type}/[pred_iam]_softmax.jpg',
            cmap='jet',
        )
    

    def plot_sigmoid_iam(self, iams, titles, save_path):
        # -----------
        # IAM Sigmoid.
        vis_preds_iams = iams.clone().sigmoid().cpu().detach().numpy()
        
        visualize_grid_v2(
            masks=vis_preds_iams[0, ...], 
            titles=titles,
            ncols=self.ncols, 
            path=f'{save_path}/{self.inst_type}/[pred_iam]_sigmoid.jpg',
            cmap='jet', # plasma
        )


    def _plot_preds(self, output, save_path):
        """
        Pred IAMs Visuals
        """
        if not f'{self.inst_type}_iams' in output['pred_iams']:
            return 
        
        iams = output['pred_iams'][f'{self.inst_type}_iams']

        probs = output['pred_logits'].softmax(-1)
        scores = probs[0, ...].cpu().detach().numpy()
        scores = np.round(scores, 3)
        titles = [', '.join([f"({class_idx}, {score:.2f})" for class_idx, score in 
                             zip(range(scores.shape[1]), score)]) for score in scores]

        self.plot_logits_iam(iams, titles, save_path)
        self.plot_softmax_iam(iams, titles, save_path)
        self.plot_sigmoid_iam(iams, titles, save_path)
        

        
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


    def _plot_iam_heads(self, cfg, masks, iams, save_path, name=''): 
        B, N, H, W = iams.shape
        groups = iams.shape[1] // cfg.model.decoder.instance_head.num_masks

        fig, axs = plt.subplots(self.nrows, groups+1, figsize=((groups+1)*2, 30))
        for i in range(self.nrows):
            for j in range(groups+1):
                ax = axs[i, j]
                ax.axis('off')

                if j == 0:
                    ax.imshow(masks[0, i, :, :], cmap='viridis')
                    ax.set_title(f'pred {i}', fontsize=10)
                else:
                    channel_idx = N // groups * (j-1) + i
                    ax.imshow(iams[0, channel_idx, :, :], cmap='jet')
                    ax.set_title(f'head {j-1}', fontsize=10)
        
        fig.tight_layout(pad=0.5)
        plt.savefig(f'{save_path}/{self.inst_type}/[pred_iam]_{name}_grouped.jpg')
        plt.close()


    def plot_logits_iam_heads(self, cfg, masks, iams, save_path):
        # -----------
        # IAM Logits [Grouped]. 
        iams = iams.clone().cpu().detach().numpy()
        self._plot_iam_heads(cfg, masks, iams, save_path, name='logits')


    def plot_softmax_iam_heads(self, cfg, masks, iams, save_path):
        # -----------
        # IAM Softmax [Grouped].  
        B, N, H, W = iams.shape
        _iam = iams.clone()
        _iam = F.softmax(_iam.view(B, N, -1), dim=-1)
        _iam = _iam.view(B, N, H, W)
        iams = _iam.cpu().detach().numpy()
        self._plot_iam_heads(cfg, masks, iams, save_path, name='softmax')


    def plot_sigmoid_iam_heads(self, cfg, masks, iams, save_path):
        # -----------
        # IAM Sigmoid [Grouped]. 
        iams = iams.clone().sigmoid().cpu().detach().numpy()
        self._plot_iam_heads(cfg, masks, iams, save_path, name='sigmoid')


    def plot_iam_heads(self, cfg, output, save_path):
        if not f'{self.inst_type}_iams' in output['pred_iams']:
            return 
        
        masks = output[f'pred_{self.inst_type}_masks'].sigmoid().cpu().detach().numpy()
        iams = output['pred_iams'][f'{self.inst_type}_iams']
        
        self.plot_logits_iam_heads(cfg, masks, iams, save_path)
        self.plot_softmax_iam_heads(cfg, masks, iams, save_path)
        self.plot_sigmoid_iam_heads(cfg, masks, iams, save_path)
