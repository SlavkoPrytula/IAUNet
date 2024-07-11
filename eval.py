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
import json

from configs import cfg as _cfg
from configs.base import dict
from models.build_model import build_model, load_model
from utils.seed import set_seed
from utils.files import increment_path

from utils.augmentations import train_transforms, valid_transforms
from utils.augmentations import normalize

from utils.evaluate import *
from utils.optimizers import *
from utils.schedulers import *
from models.seg.loss import *
from models.seg.matcher import *

from dataset.dataloaders import build_loader
from utils.registry import DATASETS, EVALUATORS
from configs.datasets import DATASETS_CFG

from visualizations.coco_vis import save_coco_vis


def load_results(file_path):
    if os.path.isfile(file_path):
        try:
            with open(file_path, 'r') as file:
                content = file.read().strip()
                if content:
                    return json.loads(content)
                else:
                    return {}
        except json.JSONDecodeError:
            return {}
    return {}

def save_results(file_path, results):
    sorted_results = {}
    for dataset_name in sorted(results):
        sorted_results[dataset_name] = {}
        for dataset_path in sorted(results[dataset_name]):
            sorted_results[dataset_name][dataset_path] = results[dataset_name][dataset_path]

    with open(file_path, 'w') as file:
        json.dump(sorted_results, file, indent=4)

def update_results(existing_results, dataset_name, new_results, dataset_path):
    if dataset_name not in existing_results:
        existing_results[dataset_name] = {}

    existing_results[dataset_name][dataset_path] = new_results
    return existing_results


def run(cfg: _cfg):
    # create directories.
    cfg.save_dir = Path(cfg.save_dir)
    print(f"Saving to {cfg.save_dir}\n")

    cfg.visuals_dir = cfg.save_dir / 'visuals'
    cfg.results_dir = cfg.save_dir / 'results'
    
    makedirs(cfg.save_dir, exist_ok=True)
    makedirs(cfg.visuals_dir, exist_ok=True)
    makedirs(cfg.results_dir, exist_ok=True)


    # set seed for reproducibility.
    set_seed(cfg.seed)


    # get dataloaders
    dataset = DATASETS.get(cfg.dataset.type)
    train_dataset = dataset(cfg, 
                            dataset_type="train", 
                            normalization=normalize, 
                            transform=valid_transforms(cfg)
                            )
    
    valid_dataset = dataset(cfg, 
                            dataset_type="valid",
                            normalization=normalize, 
                            transform=valid_transforms(cfg),
                            )
    
    train_dataloader = build_loader(train_dataset, batch_size=cfg.train.batch_size, num_workers=1, seed=cfg.seed)
    valid_dataloader = build_loader(valid_dataset, batch_size=cfg.valid.batch_size, num_workers=1, seed=cfg.seed)

    # build and prepare model.
    model = build_model(cfg)
    model.eval()

    # evaluate.
    evaluator = EVALUATORS.get(cfg.model.evaluator.type)(cfg=cfg, model=model, dataset=valid_dataset)  # from config
    evaluator(valid_dataloader)
    evaluator.evaluate(verbose=True)
    
    # save results.
    stats = evaluator.stats
    results_file = cfg.results_dir / 'evaluation_results.json'
    dataset_name = cfg.dataset.name
    dataset_path = cfg.dataset.eval_dataset.ann_file

    results = load_results(results_file)
    results = update_results(results, dataset_name, stats, dataset_path)
    save_results(results_file, results)

    
    # plot results.
    gt_coco = evaluator.gt_coco
    pred_coco = evaluator.pred_coco

    # TODO: 2config : Visualizations {n_samples: int = 5}
    # n_samples = len(valid_dataset)
    n_samples = 6
    for batch in islice(valid_dataloader, n_samples):
        targets = batch[0]
        
        img = targets["image"][0]
        fname = targets["file_name"]
        idx = targets["coco_id"]
        H, W = targets["ori_shape"]
        out_file = join(cfg.visuals_dir, f'{fname}.jpg')

        save_coco_vis(img, gt_coco, pred_coco, idx, shape=[H, W], path=out_file)


def get_config_from_path(path: str) -> _cfg:
    try:
        from models import import_from_file
        module = import_from_file(join(path, "config_files/base.py"))
        config = getattr(module, 'cfg')
    except:
        config = _cfg
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation with IAUNet')
    parser.add_argument('--experiment_name', type=str, default='', help='name of the experiment')
    return parser.parse_args()


if __name__ == '__main__':
    import sys
    sys.path.append("./")
    args = parse_args()

    experiment_path = Path("runs/[iaunet_occluders]/[ResNet]/[worms]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'overlaps', 'visible']]/[InstanceHead-v2.0-overlaps]/[job=51724647]-[2024-07-09 15:39:18]")
    cfg = get_config_from_path(experiment_path)
    old_dataset = cfg.dataset.name
    
    # cfg.dataset = "rectangle"
    cfg.dataset = "worms"
    # cfg.dataset = "brightfield"
    # cfg.dataset = "brightfield_coco"
    # cfg.dataset = "brightfield_coco_v2.0"
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
        # type="One2OneMatchingEvaluator",
        # type="IterativeEvaluator",
        mask_thr=0.5,
        score_thr=0.1,
        nms_thr=0.5,
        metric='segm', 
        classwise=True,
        outfile_prefix="results/coco",
        # max_iters=100
    )

    cfg.model.criterion.matcher=dict(
        type='HungarianMatcher',
        cost_dice=2.0,
        cost_cls=2.0,
        cost_mask=5.0
    )

    # not the cleanest way to do this
    cfg.run.run_name = join(cfg.run.run_name.replace(old_dataset, cfg.dataset.name))
    # cfg.run.run_name = '/'.join([seg for seg in cfg.run.run_name.split('/') if old_dataset not in seg])
    cfg.run.exist_ok = False
    cfg.save_dir = join(cfg.run.runs_dir, "evals", cfg.run.experiment_name, 
                        cfg.run.run_name, cfg.run.group_name, str(experiment_path).split("/")[-1])
    
    # TODO: make load_pretrained unified from loading pretrained model and model from file (weights_path)
    cfg.model.weights = experiment_path / "checkpoints/best.pth" 
    cfg.model.load_pretrained = True
    cfg.model.save_model_files = False

    cfg.valid.batch_size = 1
    cfg.train.batch_size = 1
    cfg.train.n_folds = 5
    # cfg.valid.size = [768, 768]

    # loading model from path (runs/.../[<run_name>])
    cfg.model.load_from_files = True
    cfg.model.model_files = experiment_path / "model_files"

    run(cfg)

