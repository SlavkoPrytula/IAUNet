import os
from os import mkdir, makedirs
from os.path import join
import os.path as osp

import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
from pathlib import Path
import cv2

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

from scipy.ndimage import center_of_mass
from scipy.stats import kde, gaussian_kde
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

import sys
sys.path.append("./")

from configs import cfg
from models.build_model import build_model, load_model


from utils.seed import set_seed
from utils.cuda import cuda_init

from configs.utils import save_config
from utils.files import increment_path

from utils.evaluate.dataloader_evaluator import DataloaderEvaluator
from utils.coco.coco import COCO

from utils.utils import nested_tensor_from_tensor_list, flatten_mask
from utils.visualise import visualize_grid_v2, visualize

from utils.augmentations import train_transforms, valid_transforms
from utils.normalize import normalize

from utils.optimizers import *
from utils.schedulers import *
from models.seg.loss import *
from models.seg.matcher import *

from dataset.dataloaders import build_loader
from utils.registry import DATASETS

from configs.datasets import DATASETS_CFG

from utils.metrics.saliency import NSS, AUC_Borji, CrossEntropy


def calculate_center_of_mass(mask):
    return center_of_mass(mask)


palette = [
    (1, 0, 0, 1),    # Red
    (0, 1, 0, 1),    # Green
    (0, 0, 1, 1),    # Blue
    (1, 1, 0, 1),    # Yellow
    (1, 0.4, 0.7, 1),# Magenta
    (0, 1, 1, 1),    # Cyan
    (1, 0.5, 0, 1),  # Orange
    (0.5, 0, 0.8, 1),# Purple
    (0, 0.75, 0, 1), # Bright Green
    (0.85, 0.85, 0, 1), # Olive
    (0.7, 0, 0.5, 1),# Maroon
    (0, 0.7, 0.7, 1),# Teal
    (0.9, 0.3, 0.5, 1),# Pink
    (0.3, 0.8, 0.6, 1),# Sea Green
    (0.6, 0.3, 0.9, 1),# Lavender
    (0.3, 0.6, 0.9, 1),# Sky Blue
    (0.8, 0.7, 0.2, 1),# Sand
    (1, 0.65, 0, 1), # Bright Orange
    (0.4, 0.9, 0.4, 1),# Light Green
    (0, 0.8, 0.8, 1),# Aquamarine
    (0.6, 0.2, 0.9, 1),# Indigo
    (1, 0.2, 0.4, 1),# Hot Pink
    (0.4, 0.4, 0.4, 1),# Dark Grey
    (0.8, 0.3, 0, 1), # Rust
    (0.5, 0.7, 0.3, 1) # Moss Green
]

def run(cfg: cfg):
    set_seed(cfg.seed)
    
    # get dataloaders
    dataset = DATASETS.get(cfg.dataset.type)
    valid_dataset = dataset(cfg, 
                            dataset_type="train",
                            normalization=normalize, 
                            transform=valid_transforms(cfg)
                            )
    
    valid_dataloader = build_loader(valid_dataset, batch_size=cfg.valid.batch_size, num_workers=2)

    # build and prepare model
    model = build_model(cfg)
    model.eval()

    offsets = []
    iam_thr = 0.95

    for step, batch in enumerate(valid_dataloader):
        if batch is None:
            continue
        
        # prepare targets
        images = []
        targets = []

        # for target in batch:
        target = batch[0]
        target = {k: v.to(cfg.device) for k, v in target.items()}
        images.append(target["image"])
        targets.append(target)

        image = nested_tensor_from_tensor_list(images)   # (B, C, H, W)

        # predict.
        with torch.no_grad():
            results = model(image.tensors[:, :-1, :, :])

        scores = results['pred_logits'].sigmoid()
        scores = scores[0, :, 0]

        mask_preds = results['pred_masks'].sigmoid()
        mask_preds = mask_preds[0, ...]

        iam_preds = results['pred_iam']['iam']
        iam_preds = iam_preds[0, ...]

        # seg_masks = masks_pred > self.mask_threshold
        # sum_masks = seg_masks.sum((1, 2)).float()

        # # maskness scores.
        # maskness_scores = (masks_pred * seg_masks.float()).sum((1, 2)) / sum_masks
        # scores = maskness_scores

        # scores = scores.detach().cpu().numpy()
        # masks_pred = masks_pred.detach().cpu().numpy()
        # masks_pred = (masks_pred > cfg.model.).astype(np.uint8)

        masks = target['masks']
        masks = masks.detach().cpu().numpy()

        images = target["image"]
        images = images.detach().cpu().numpy()


        # np.save()
        N, H, W = iam_preds.shape
        iam_preds = iam_preds.sigmoid()
        
        # plot - iam - aggregated
        fig, ax = plt.subplots(1, 1, figsize=[30, 30])
        fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        agg_iam = np.sum([iam.cpu().detach().numpy() for iam, _ in zip(iam_preds, mask_preds)], axis=0)
        agg_iam_norm = agg_iam / np.max(agg_iam)  # Normalize

        # Plot the aggregate IAM with jet colormap
        # ax.imshow(agg_iam_norm, cmap='jet')
        background_jet = plt.cm.jet(agg_iam_norm)[:, :, :3] 
        cmap = plt.cm.get_cmap('hsv', len(iam_preds))

        # ax.imshow(np.zeros((H, W)), cmap='gray')
        # ax.imshow(images[0, ...], cmap='gray')

        # Overlay each IAM with vibrant colors
        # for i, (iam, mask) in enumerate(zip(iam_preds, masks)):
        #     iam = iam.cpu().detach().numpy()
        #     iam_norm = iam / np.max(iam)  # Normalize IAM

        #     # Select color from the vibrant_colors list, cycling if necessary
        #     color = palette[i % len(palette)]

        #     # Create a colored IAM using the selected color
        #     colored_iam = np.zeros((iam.shape[0], iam.shape[1], 4))  # 4 for RGBA
        #     for j in range(3):  # RGB channels
        #         colored_iam[..., j] = cmap(i)[j] #color[j] 
        #     colored_iam[..., 3] = iam_norm  # Alpha channel

        #     # Overlay the colored IAM
        #     ax.imshow(colored_iam, alpha=1)

        # ax.axis('off')
        # plt.tight_layout()
        # plt.savefig(f"./tools/analysis/iams/visuals/pred_{step}.jpg")
        # if step == 0:
        #     raise
        
        # agg_iam = np.zeros((512, 512))
        # highlight_idx = 10
        
        # for i, (iam, mask) in enumerate(zip(iam_preds, mask_preds)):
        #     iam = iam.cpu().detach().numpy()
        #     mask = mask.cpu().detach().numpy()

        #     iam /= N

        #     if i != highlight_idx:
        #         agg_iam += iam * 0.3
        #     else:
        #         agg_iam += iam

        # ax.imshow(agg_iam, cmap='jet')

        ax.imshow(np.zeros((H, W)), cmap='gray')
        # agg_iam_rgba = np.zeros((512, 512, 4))
        highlight_idx = 11

        # agg_iam = np.zeros((512, 512))

        # # First, plot all IAMs with reduced alpha
        for i, iam in enumerate(iam_preds):
            if i != highlight_idx:
                iam = iam.cpu().detach().numpy()
                # agg_iam += iam    
                colored_iam = np.zeros((iam.shape[0], iam.shape[1], 4))  # 4 for RGBA
                for j in range(3):  # RGB channels
                    colored_iam[..., j] = cmap(i)[j] #color[j] 
                colored_iam[..., 3] = iam  # Alpha channel

                ax.imshow(colored_iam, alpha=0.15)

        # ax.imshow(agg_iam, alpha=0.5)

        # Overlay the highlighted IAM
        highlight_iam = iam_preds[highlight_idx].cpu().detach().numpy()
        highlight_iam_norm = highlight_iam / np.max(highlight_iam)



        # Custom colormap for the highlighted IAM
        # highlight_iam_colored = plt.cm.jet(highlight_iam_norm)
        highlight_iam_colored = np.zeros((iam.shape[0], iam.shape[1], 4))  # 4 for RGBA
        for j in range(3):  # RGB channels
            highlight_iam_colored[..., j] = cmap(i)[j] #color[j] 
        highlight_iam_colored[..., 3] = iam  # Alpha channel

        # Set lower values to be more transparent
        highlight_iam_colored[..., 3] = np.clip(highlight_iam_norm * 3, 0, 1)


        binary_mask = highlight_iam_norm > 0.15

        # Apply Sobel filter to find edges of the binary mask
        sobel_x = ndimage.sobel(binary_mask, axis=0)
        sobel_y = ndimage.sobel(binary_mask, axis=1)
        sobel_magnitude = np.hypot(sobel_x, sobel_y)

        # Create a binary mask for the contour
        # contour_mask = sobel_magnitude > 0  # This creates a mask for the edges
        # contour_threshold = np.percentile(sobel_magnitude, 95)  # Adjust this percentile as needed
        contour_mask = sobel_magnitude > 0
        # contour_mask = binary_erosion(contour_mask, iterations=1)
        # structuring_element = generate_binary_structure(1, 1)  # 2D, radius=1 (circular)
        # contour_mask = binary_erosion(contour_mask, structure=structuring_element, iterations=1)


        # Create RGBA image for the contour
        contour_rgba = np.zeros_like(highlight_iam_colored)
        contour_rgba[..., 3] = contour_mask  # Set alpha channel to the binary mask
        contour_rgba[contour_mask, :3] = 1  # Set color to white for the contour
        

        # Display the images
        ax.imshow(highlight_iam_colored)
        ax.imshow(contour_rgba)


        ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"./tools/analysis/iams/visuals/pred_{step}.jpg", dpi=300)
        raise



        # ax.imshow(np.zeros((H, W)), cmap='gray')

        # # List of colormaps to use for non-highlighted IAMs
        # colormaps = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges', 'Greys']

        # # Plot all non-highlighted IAMs with different colormaps and reduced alpha
        # for i, iam in enumerate(iam_preds):
        #     if i != highlight_idx:
        #         iam = iam.cpu().detach().numpy()
        #         iam_norm = iam / np.max(iam)  # Normalize IAM
        #         cmap = colormaps[i % len(colormaps)]  # Cycle through the colormap list
        #         ax.imshow(iam_norm, cmap=cmap, alpha=0.5)

        # # Overlay the highlighted IAM with the jet colormap and custom transparency
        # highlight_iam = iam_preds[highlight_idx].cpu().detach().numpy()
        # highlight_iam_norm = highlight_iam / np.max(highlight_iam)

        # # Create an RGBA version of the highlighted IAM for custom alpha manipulation
        # highlight_iam_rgba = plt.cm.jet(highlight_iam_norm)  # Apply jet colormap

        # # Adjust alpha based on intensity (e.g., higher alpha for higher values)
        # alpha_threshold = 0.1  # Threshold for starting to show the iam
        # highlight_iam_rgba[..., 3] = np.clip((highlight_iam_norm - alpha_threshold) * 10, 0, 1)

        # # Overlay the highlighted IAM with adjusted alpha
        # ax.imshow(highlight_iam_rgba)


        # ax.axis('off')
        # plt.tight_layout()
        # plt.savefig(f"./tools/analysis/iams/visuals/pred_{step}.jpg")
        # raise



# model.eval()

# for step, batch in enumerate(dataloader):
#     if batch is None:
#         continue
    
#     # prepare targets
#     images = []
#     targets = []

#     # for target in batch:
#     target = batch[0]
#     target = {k: v.to(cfg.device) for k, v in target.items()}
#     images.append(target["image"])
#     targets.append(target)

#     image = nested_tensor_from_tensor_list(images)   # (B, C, H, W)

#     # predict.
#     output = self.inference_single(model, image.tensors)

#     pred = output
#     scores = pred['pred_logits'].sigmoid()
#     scores = scores[0, :, 0]

#     masks_pred = pred['pred_masks'].sigmoid()
#     masks_pred = masks_pred[0, ...]

#     seg_masks = masks_pred > self.mask_threshold
#     sum_masks = seg_masks.sum((1, 2)).float()

#     # maskness scores.
#     maskness_scores = (masks_pred * seg_masks.float()).sum((1, 2)) / sum_masks
#     scores = maskness_scores

#     # masks_pred = masks_pred[scores > 0.4]
#     # scores = scores[scores > 0.4]

#     scores = scores.detach().cpu().numpy()
#     masks_pred = masks_pred.detach().cpu().numpy()
#     masks_pred = (masks_pred > self.mask_threshold).astype(np.uint8)

#     masks = target['masks']
#     masks = masks.detach().cpu().numpy()



if __name__ == '__main__':
    # ablation studies | kernel size ✅ 
    # new - [ms]
    # kernel_size=128
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[softmax_iam]/[kernel_dim=128]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49177327]-[2023-11-12 13:49:27]")
    # kernel_size=256
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49177916]-[2023-11-12 14:19:32]")
    # kernel_size=512
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[softmax_iam]/[kernel_dim=512]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49177398]-[2023-11-12 13:53:19]")

    # ablation studies | activation functions ✅
    # softmax
    # - same as [kernel_size=256]
    # temp_softmax
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[temp_softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49179477]-[2023-11-13 03:33:38]")
    # sigmoid
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[sigmoid_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49179772]-[2023-11-13 09:53:49]")

    
    
    # experiment_path = Path("runs/[sparse_seunet]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=45867499]-[2023-08-07 11:44:24]")
    experiment_path = Path("runs/[sparse_seunet]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=47344335]-[2023-08-30 18:08:43]")
    # experiment_path = Path("runs/[sparse_seunet]-[512]/[synthetic_brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=46860262]-[2023-08-21 15:48:36]")
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[sigmoid_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=48862144]-[2023-10-29 01:10:11]")



    old_dataset = cfg.dataset.name
    
    cfg.dataset = "brightfield"
    # cfg.dataset = "EVICAN2Easy"
    # cfg.dataset = "EVICAN2Medium"
    # cfg.dataset = "EVICAN2Difficult"
    # cfg.dataset = "LiveCell30Images"
    cfg.dataset = DATASETS_CFG.get(cfg.dataset)
    
    # model params
    cfg.model.in_channels = 2
    cfg.model.kernel_dim = 128
    cfg.model.num_masks = 50

    cfg.run.run_name = join(cfg.run.run_name.replace(old_dataset, cfg.dataset.name))
    cfg.run.exist_ok = False
    
    # TODO: make load_pretrained unified from loading pretrained model and model from file (weights_path)
    cfg.model.weights = experiment_path / "checkpoints/best.pth" 
    cfg.model.load_pretrained = True
    cfg.model.save_model_files = False

    cfg.valid.batch_size = 1
    cfg.train.batch_size = 1
    cfg.train.n_folds = 5

    # loading model from path (runs/.../[<run_name>])
    cfg.model.load_from_files = True
    cfg.model.model_files = experiment_path / "model_files"

    cfg.run.save_path = "./tools/analysis/iams"

    run(cfg)