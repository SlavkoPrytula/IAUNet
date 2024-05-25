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

from configs import cfg
from configs.base import dict
from models.build_model import build_model, load_model
from utils.seed import set_seed
from utils.files import increment_path

from utils.augmentations import train_transforms, valid_transforms
from utils.normalize import normalize
from utils.evaluate import DataloaderEvaluator, MMDetDataloaderEvaluator

from utils.optimizers import *
from utils.schedulers import *
from models.seg.loss import *
from models.seg.matcher import *

from dataset.dataloaders import build_loader
from utils.registry import DATASETS, EVALUATORS
from configs.datasets import DATASETS_CFG


def run(cfg: cfg):
    # create directories.
    cfg.save_dir = increment_path(
        join(cfg.run.runs_dir, "evals", cfg.run.experiment_name, cfg.run.run_name), 
        exist_ok=cfg.run.exist_ok
        )
    print(cfg.save_dir)

    cfg.visuals_dir = cfg.save_dir / 'visuals'
    makedirs(cfg.visuals_dir, exist_ok=True)
    makedirs(cfg.save_dir / 'results', exist_ok=True)

    # set seed for reproducibility
    set_seed(cfg.seed)
    

    # get dataloaders
    dataset = DATASETS.get(cfg.dataset.type)
    train_dataset = dataset(cfg, 
                            dataset_type="train", 
                            normalization=normalize, 
                            transform=valid_transforms(cfg)
                            )
    # valid_dataset = dataset(cfg, 
    #                         dataset_type="valid",
    #                         normalization=normalize, 
    #                         transform=valid_transforms(cfg)
    #                         )
    
    valid_dataset = dataset(cfg, 
                            dataset_type="eval",
                            normalization=normalize, 
                            transform=valid_transforms(cfg)
                            )
    
    train_dataloader = build_loader(train_dataset, batch_size=cfg.train.batch_size, num_workers=2)
    valid_dataloader = build_loader(valid_dataset, batch_size=cfg.valid.batch_size, num_workers=2)

    # build and prepare model
    model = build_model(cfg)
    model.eval()

    # evaluate.
    evaluator = EVALUATORS.get(cfg.model.evaluator.type)(cfg=cfg, dataset=valid_dataset)  # from config
    evaluator(model, valid_dataloader)
    evaluator.evaluate(verbose=True)

    # plot results.
    gt_coco = evaluator.gt_coco
    pred_coco = evaluator.pred_coco

    for i in range(1, 5):
        targets = valid_dataset[i-1]
        idx = valid_dataset.image_ids[i-1]
        img_path = targets["img_path"]
        fname, name = osp.splitext(osp.basename(img_path))
        out_file = join(cfg.visuals_dir, f'{fname}.jpg')

        H, W = targets["ori_shape"]
        img = targets["image"][0]

        fig, ax = plt.subplots(1, 2, figsize=[20, 10])
        fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        # img1
        annIds  = gt_coco.getAnnIds(imgIds=[idx])
        anns    = gt_coco.loadAnns(annIds)
        ax[0].imshow(img, cmap='gray')

        gt_masks = gt_coco.getMasks(anns, alpha=0.3)
        for gt_mask in gt_masks:
            gt_mask = cv2.resize(gt_mask, (W, H))
            ax[0].imshow(gt_mask)

        # img2
        annIds  = pred_coco.getAnnIds(imgIds=[idx])
        anns    = pred_coco.loadAnns(annIds)
        ax[1].imshow(img, cmap='gray')

        pred_masks = gt_coco.getMasks(anns, alpha=0.3)
        for pred_mask in pred_masks:
            pred_mask = cv2.resize(pred_mask, (W, H))
            ax[1].imshow(pred_mask)


        for a in ax:
            a.axis('off')
            a.set_xlim(0, W)
            a.set_ylim(H, 0)

        fig.canvas.draw()
        fig.savefig(out_file, bbox_inches='tight', pad_inches=0)
        plt.close(fig)



def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation with SparseUnet')
    parser.add_argument('--experiment_name', type=str, default='', help='name of the experiment')
    return parser.parse_args()


if __name__ == '__main__':
    import sys
    sys.path.append("./")
    args = parse_args()


    experiment_path = Path("runs/[sparse_seunet]/[brightfield_coco_v2.0]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=50263497]-[2024-02-02 11:34:13]")
    old_dataset = cfg.dataset.name
    
    # cfg.dataset = "brightfield"
    # cfg.dataset = "brightfield_coco"
    cfg.dataset = "brightfield_coco_v2.0"
    # cfg.dataset = "EVICAN2Easy"
    # cfg.dataset = "EVICAN2Medium"
    # cfg.dataset = "EVICAN2Difficult"
    # cfg.dataset = "LiveCell"
    # cfg.dataset = "LiveCell30Images"
    # cfg.dataset = "YeastNet"
    # cfg.dataset = "HuBMAP"
    cfg.dataset = DATASETS_CFG.get(cfg.dataset)
    
    # model params
    # cfg.model.in_channels = 1
    cfg.model.num_masks = 100
    cfg.model.kernel_dim = 256
    cfg.model.mask_dim = 256

    cfg.model.evaluator=dict(
        type="MMDetDataloaderEvaluator",
        # type="AnalysisDataloaderEvaluator",
        mask_thr=0.1,
        cls_thr=-1,
        score_thr=0.05,
        nms_thr=0.5,
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

