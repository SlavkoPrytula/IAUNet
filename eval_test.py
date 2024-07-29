import os
from os import mkdir, makedirs
from os.path import join
import os.path as osp

import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import argparse
from tqdm import tqdm
from itertools import islice

from configs import cfg as _cfg
from configs.base import dict
from models.build_model import build_model, load_model
from utils.seed import set_seed
from utils.files import increment_path

from utils.augmentations import train_transforms, valid_transforms
from utils.normalize import normalize

from utils.evaluate import *
from utils.optimizers import *
from utils.schedulers import *
from models.seg.loss import *
from models.seg.matcher import *

from dataset.dataloaders import build_loader
from utils.registry import DATASETS, EVALUATORS
from configs.datasets import DATASETS_CFG

from visualizations.coco_vis import save_coco_vis


def run(cfg: _cfg):
    # create directories.
    # cfg.save_dir = increment_path(
    #     join(cfg.run.runs_dir, "evals", cfg.run.experiment_name, cfg.run.run_name), 
    #     exist_ok=cfg.run.exist_ok
    #     )
    # print(cfg.save_dir)

    # cfg.visuals_dir = cfg.save_dir / 'visuals'
    # makedirs(cfg.visuals_dir, exist_ok=True)

    # # set seed for reproducibility.
    # set_seed(cfg.seed)
    

    # get dataloaders
    dataset = DATASETS.get(cfg.dataset.type)
    train_dataset = dataset(cfg, 
                            dataset_type="train", 
                            normalization=normalize, 
                            transform=valid_transforms(cfg)
                            )
    
    valid_dataset = dataset(cfg, 
                            dataset_type="eval",
                            normalization=normalize, 
                            transform=valid_transforms(cfg)
                            )
    
    train_dataloader = build_loader(train_dataset, batch_size=cfg.train.batch_size, num_workers=1)
    valid_dataloader = build_loader(valid_dataset, batch_size=cfg.valid.batch_size, num_workers=1)

    # build and prepare model.
    model = build_model(cfg)
    model.eval()

    for step, batch in enumerate(valid_dataloader):
        if batch is None:
            continue
        
        # prepare targets
        images = []
        targets = []
        for i in range(len(batch)):
            target = batch[i]

            ignore = ["img_id", "img_path", "ori_shape", "file_name", "coco_id"]
            target = {k: v.to(cfg.device) if k not in ignore else v 
                    for k, v in target.items()}
            images.append(target["image"])

            targets.append(target)
            
        images = nested_tensor_from_tensor_list(images)   # (B, C, H, W)
        batch_size = images.tensors.size(0)
        
        output = model(images.tensors)


        # ids = [0, 2, 3]
        ids = np.arange(5)
        # -----------
        # Pred Masks.
        vis_preds_cyto = output['pred_masks'].sigmoid().cpu().detach().numpy()
        probs = output['pred_logits'].softmax(-1)
        scores = probs[0, :, 0].cpu().detach().numpy()
        scores = np.round(scores, 2)

        iou_scores = output['pred_scores'].sigmoid()
        iou_scores = iou_scores[0, :, 0].cpu().detach().numpy()
        iou_scores = np.round(iou_scores, 2)

        titles = [
            f"conf: {score:.2f}, iou: {iou_score:.2f}"
            for score, iou_score in zip(scores, iou_scores)
        ]
        
        visualize_grid_v2(
            figsize=[10, 5],
            masks=vis_preds_cyto[0, ...][ids], 
            titles=np.array(titles)[ids], 
            ncols=5, 
            nrows=1, 
            path=f'[pred_masks].jpg'
        )


        # -----------
        # IAMs
        iam = output['pred_iam']
        B, N, H, W = iam.shape

        probs = output['pred_logits'].softmax(-1)
        scores = probs[0, ...].cpu().detach().numpy()
        scores = np.round(scores, 3)
        titles = [', '.join([f"({class_idx}, {score:.2f})" for class_idx, score in 
                             zip(range(scores.shape[1]), score)]) for score in scores]

        # -----------
        # IAM Logits. 
        vis_preds_iams = iam.clone().cpu().detach().numpy()
        
        visualize_grid_v2(
            figsize=[10, 5],
            masks=vis_preds_iams[0, ...][ids], 
            titles=np.array(titles)[ids],
            ncols=5, 
            nrows=1, 
            path=f'[pred_iam]_logits.jpg',
            cmap='jet',
        )

        # -----------
        # IAM Softmax.
        _iam = iam.clone()
        _iam = F.softmax(_iam.view(B, N, -1), dim=-1)
        _iam = _iam.view(B, N, H, W)
        vis_preds_iams = _iam.cpu().detach().numpy()
        
        visualize_grid_v2(
            figsize=[10, 5],
            masks=vis_preds_iams[0, ...][ids], 
            titles=np.array(titles)[ids],
            ncols=5, 
            nrows=1, 
            path=f'[pred_iam]_softmax.jpg',
            cmap='jet',
            # vmin=0, vmax=1
        )

        raise


def get_config_from_path(path: str) -> _cfg:
    try:
        from models import import_from_file
        module = import_from_file(join(path, "config_files/base.py"))
        config = getattr(module, 'cfg')
    except:
        config = _cfg
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation with SparseUnet')
    parser.add_argument('--experiment_name', type=str, default='', help='name of the experiment')
    return parser.parse_args()


if __name__ == '__main__':
    import sys
    sys.path.append("./")
    args = parse_args()

    experiment_path = Path("runs/[iaunet_optim_v2]/[experimental]/[job=50473884]-[2024-03-04 09:15:27]")
    cfg = get_config_from_path(experiment_path)
    old_dataset = cfg.dataset.name
    
    # cfg.dataset = "brightfield"
    # cfg.dataset = "brightfield_coco"
    cfg.dataset = "brightfield_coco_v2.0"
    # cfg.dataset = "EVICAN2Easy"
    # cfg.dataset = "EVICAN2Medium"
    # cfg.dataset = "EVICAN2Difficult"
    # cfg.dataset = "LiveCell"
    # cfg.dataset = "LiveCell2Percent"
    # cfg.dataset = "LiveCell30Images"
    # cfg.dataset = "YeastNet"
    # cfg.dataset = "HuBMAP"
    cfg.dataset = DATASETS_CFG.get(cfg.dataset)
    
    # model params
    cfg.model.evaluator=dict(
        type="MMDetDataloaderEvaluator",
        # type="AnalysisDataloaderEvaluator",
        mask_thr=0.5,
        score_thr=0.2,
        nms_thr=0.5,
        metric='segm', 
        classwise=True,
        outfile_prefix="results/coco"
    )

    cfg.model.criterion.matcher=dict(
        type='HungarianMatcher',
        cost_dice=5.0,
        cost_cls=2.0,
        cost_mask=5.0
    )


    cfg.run.run_name = join(cfg.run.run_name.replace(old_dataset, cfg.dataset.name), args.experiment_name)
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

    run(cfg)

