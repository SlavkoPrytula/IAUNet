import numpy as np
import matplotlib.pyplot as plt

from .base_visualizer import BaseVisualizer
from utils.registry import CALLBACKS


@CALLBACKS.register(name="AlignmentVisualizer")
class AlignmentVisualizer(BaseVisualizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot(self, cfg, output, save_path):
        self.score_iou(cfg, output, save_path)

    def score_iou(self, cfg, output, save_path):
        probs = output['pred_logits'].softmax(-1)
        scores = probs[0, :, 0].cpu().detach().numpy()
        scores = np.round(scores, 2)

        iou_scores = output['pred_scores'].sigmoid()
        iou_scores = iou_scores[0, :, 0].cpu().detach().numpy()
        iou_scores = np.round(iou_scores, 2)

        plt.figure(figsize=[10, 10])
        plt.scatter(iou_scores, scores, color="red", s=175, alpha=0.5)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("Scores", fontsize=24)
        plt.xlabel("IoU", fontsize=24)
        plt.grid(True, alpha=0.75)
        plt.tight_layout()
        plt.savefig(f'{save_path}/iou_alignment.jpg')
        plt.close()
