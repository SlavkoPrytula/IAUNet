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

from utils.metrics.saliency import NSS, AUC_Borji, CrossEntropy, MCC


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

    nss_scores = []
    bce_scores = []
    mcc_scores = []

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
            # results = model(image.tensors[:, :-1, :, :])
            results = model(image.tensors)

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
        
        # pot - iam - aggregated
        fig, ax = plt.subplots(1, 2, figsize=[20, 10])
        fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        
        # random_iam = np.random.rand(N, H, W)
        for i, (iam, mask) in enumerate(zip(iam_preds, mask_preds)):
            # iam = iam.cpu().detach().numpy()
            iam = np.random.rand(H, W)
            mask = mask.cpu().detach().numpy()

            # alignment
            nss_score = NSS(iam, mask)
            bce_score = CrossEntropy(iam, mask)
            # mcc_score = MCC(iam, mask)

            if not (np.isnan(nss_score).any() or 
                    np.isnan(bce_score).any()):

                # Store the scores
                nss_scores.append(nss_score)
                bce_scores.append(bce_score)
                # mcc_scores.append(mcc_score)

    mean_nss = np.mean(nss_scores)
    mean_bce = np.mean(bce_scores)
    # mean_mcc = np.mean(mcc_scores)

    print("Mean NSS:", mean_nss)
    print("Mean BCE:", mean_bce)
    # print("Mean MCC:", mean_mcc)



if __name__ == '__main__':
    # ablation studies | kernel size ✅ 
    # new - [ms]
    # kernel_size=128
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[softmax_iam]/[kernel_dim=128]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49177327]-[2023-11-12 13:49:27]")
    # kernel_size=256
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49177916]-[2023-11-12 14:19:32]")
    # kernel_size=512
    experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[softmax_iam]/[kernel_dim=512]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49177398]-[2023-11-12 13:53:19]")

    # ablation studies | activation functions ✅
    # softmax
    # - same as [kernel_size=256]
    # temp_softmax
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[temp_softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49179477]-[2023-11-13 03:33:38]")
    # sigmoid
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[sigmoid_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49179772]-[2023-11-13 09:53:49]")

    
    
    # experiment_path = Path("runs/[sparse_seunet]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=45966022]-[2023-08-08 17:15:00]")
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[sigmoid_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=48862144]-[2023-10-29 01:10:11]")



    old_dataset = cfg.dataset.name
    
    cfg.dataset = "brightfield"
    # cfg.dataset = "EVICAN2Easy"
    # cfg.dataset = "EVICAN2Medium"
    # cfg.dataset = "EVICAN2Difficult"
    # cfg.dataset = "LiveCell30Images"
    cfg.dataset = DATASETS_CFG.get(cfg.dataset)
    
    # model params
    cfg.model.in_channels = 3
    cfg.model.kernel_dim = 512

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