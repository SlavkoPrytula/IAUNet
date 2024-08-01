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

import sys
sys.path.append("./")

from configs import cfg
from models.build_model import build_model, load_model


from utils.seed import set_seed

from configs.utils import save_config
from utils.files import increment_path

from utils.evaluate.dataloader_evaluator import DataloaderEvaluator
from utils.coco.coco import COCO

from utils.utils import nested_tensor_from_tensor_list, flatten_mask
from utils.visualise import visualize_grid_v2, visualize

from utils.augmentations import train_transforms, valid_transforms
from utils.augmentations import normalize

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


def run(cfg: cfg):
    set_seed(cfg.seed)
    
    # get dataloaders
    dataset = DATASETS.get(cfg.dataset.type)
    valid_dataset = dataset(cfg, 
                            dataset_type="eval",
                            normalization=normalize, 
                            transform=valid_transforms(cfg)
                            )
    
    valid_dataloader = build_loader(valid_dataset, batch_size=cfg.valid.batch_size, num_workers=2)

    # build and prepare model
    model = build_model(cfg)
    model.eval()

    offsets = []
    iam_thr = 0.5

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
            results = model(image.tensors[:, :, :, :])

        scores = results['pred_logits'].sigmoid()
        scores = scores[0, :, 0]

        mask_preds = results['pred_masks'].sigmoid()
        mask_preds = mask_preds[0, ...]

        iam_preds = results['pred_iam']#['iam']
        iam_preds = iam_preds[0, ...]

        masks = target['masks']
        masks = masks.detach().cpu().numpy()


        # np.save()
        N, H, W = iam_preds.shape
        iam_preds = iam_preds.sigmoid()
        iam_preds = F.softmax(iam_preds.view(N, -1), dim=-1)
        iam_preds = iam_preds.view(N, H, W)
        
        # pot - iam - aggregated
        # fig, ax = plt.subplots(1, 2, figsize=[20, 10])
        # fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        
        for i, (iam, mask) in enumerate(zip(iam_preds, mask_preds)):
            iam = iam.cpu().detach().numpy()
            mask = mask.cpu().detach().numpy()

            # offsets
            center_iam = calculate_center_of_mass(iam > iam_thr)
            center_mask = calculate_center_of_mass(mask)
            print(center_iam, center_mask)
            # print()

            if not (np.isnan(center_iam).any() or np.isnan(center_mask).any()):
                offset_x = center_iam[1] - center_mask[1]  # x offset
                offset_y = center_iam[0] - center_mask[0]  # y offset
                offsets.append((offset_x, offset_y))


    # plot offsets
    offsets = np.array(offsets)
    distances = np.sqrt(offsets[:, 0]**2 + offsets[:, 1]**2)

    x = offsets[:, 0]
    y = offsets[:, 1]

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Normalize the distance and density for color and alpha mapping
    distance_norm = (distances - distances.min()) / (distances.max() - distances.min())
    density_norm = (z - z.min()) / (z.max() - z.min())

    # Rescale density_norm to range from 0.25 to 1.0
    density_rescaled = 0.5 * density_norm + 0.5


    plt.figure(figsize=[10, 10])
    plt.scatter(x, y, c=distance_norm, cmap='jet_r', alpha=density_rescaled, s=400, 
                edgecolor="white", linewidth=2.)

    # plt.figure(figsize=[10, 10])
    # plt.scatter(offsets[:, 0], offsets[:, 1], c=distances, cmap='plasma', 
    #             alpha=1, s=200, zorder=2)
    
    plt.axhline(0, color='red', linestyle='--', linewidth=4., alpha=0.75, zorder=1)  # Horizontal line at y=0
    plt.axvline(0, color='red', linestyle='--', linewidth=4., alpha=0.75, zorder=1)  # Vertical line at x=0
    
    plt.grid(True, which='both', linestyle='-', linewidth=2., alpha=0.5)
    # plt.xlim(-np.max(np.abs(offsets)), np.max(np.abs(offsets)))  # Set limits for x-axis
    # plt.ylim(-np.max(np.abs(offsets)), np.max(np.abs(offsets)))  # Set limits for y-axis
    plt.xlim([-256, 256])  # Set limits for x-axis
    plt.ylim([-256, 256])  # Set limits for y-axis

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(f"./tools/analysis/iams/visuals/offsets/offsets-[iam_thr@{iam_thr}].jpg")
    # plt.savefig(f"./tools/analysis/iams/visuals/offsets.jpg")


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

    
    
    # experiment_path = Path("runs/[sparse_seunet]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=45966022]-[2023-08-08 17:15:00]")
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[sigmoid_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=48862144]-[2023-10-29 01:10:11]")


    # ======================== HuBMAP =========================
    # hubmap
    experiment_path = Path("runs/[sparse_seunet]/[HuBMAP]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49189226]-[2023-11-14 02:54:30]")



    old_dataset = cfg.dataset.name
    
    # cfg.dataset = "brightfield"
    # cfg.dataset = "EVICAN2Easy"
    # cfg.dataset = "EVICAN2Medium"
    # cfg.dataset = "EVICAN2Difficult"
    # cfg.dataset = "LiveCell30Images"
    cfg.dataset = "HuBMAP"
    cfg.dataset = DATASETS_CFG.get(cfg.dataset)
    
    # model params
    cfg.model.in_channels = 3
    cfg.model.kernel_dim = 256

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