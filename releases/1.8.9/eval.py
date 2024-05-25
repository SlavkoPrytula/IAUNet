import os
from os import mkdir, makedirs
from os.path import join
import gc
import importlib.util

import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from configs import cfg
from dataset.dataloaders import get_dataloaders
from models.build_model import build_model, load_model
from dataset.prepare_dataset import get_folds

# from dataset.datasets.brightfiled import df as _df
from dataset.datasets.rectangle import df as _df

from utils.seed import set_seed
from utils.cuda import cuda_init

from configs.utils import save_config
from utils.files import increment_path

from utils.evaluate.dataloader_evaluator import DataloaderEvaluator
from utils.coco.coco import COCO

from utils.utils import nested_tensor_from_tensor_list, flatten_mask
from utils.visualise import visualize_grid_v2, visualize

import argparse
from tqdm import tqdm


def run(cfg: cfg):
    # create directories.
    cfg.save_dir = increment_path(
        join(cfg.run.runs_dir, "evals", cfg.run.experiment_name, cfg.run.run_name), 
        exist_ok=cfg.run.exist_ok
        )
    print(cfg.save_dir)

    cfg.visuals_dir = cfg.save_dir / 'visuals'
    makedirs(cfg.visuals_dir, exist_ok=True)

    # set seed for reproducibility
    set_seed(cfg.seed)

    # 5-fold split
    df = get_folds(cfg, _df)
    print(df.groupby(['fold', 'cell_line'])['id'].count())
    

    # Run training
    for fold_i in [0]:
        print(f'+ Fold: {fold_i}')
        print(f'-' * 10)
        print()

        # get dataloaders
        train_loader, valid_loader = get_dataloaders(cfg, df, fold=fold_i)

        # build and prepare model
        model = load_model(cfg, cfg.model.weights)
        # model.eval()

        # TODO: Class for plotting

        # evaluate.
        evaluator = DataloaderEvaluator(cfg=cfg)
        evaluator(model, valid_loader)
        evaluator.evaluate()

        print(evaluator.coco_eval.stats)


def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation with SparseUnet')
    parser.add_argument('--experiment_name', type=str, default='', help='name of the experiment')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=45867499]-[2023-08-07 11:44:24]")

    config_path = experiment_path / "default.yaml"
    cfg.yaml_load(config_path)

    cfg.run.run_name = join(cfg.run.run_name, args.experiment_name)
    cfg.run.exist_ok = False

    # changing datasets
    # cfg.dataset.coco_dataset = join(cfg.project.home_dir, f'data/datasets/synthetic_datasets/rectangle/rectangles_[n=100]_[R_min=2_R_max=15]_[06.08.23].json')
    # cfg.dataset.coco_dataset = join(cfg.project.home_dir, f'data/datasets/synthetic_datasets/rectangle/rectangles_[n=100]_[R_min=2_R_max=5]_[06.08.23].json')
    
    # TODO: make load_pretrained unified from loading pretrained model and model from file (weights_path) - done
    cfg.model.weights = experiment_path / "checkpoints/best.pth"
    cfg.model.load_pretrained = True
    cfg.model.save_model_files = False

    cfg.valid.bs = 1
    cfg.train.bs = 1
    cfg.train.n_folds = 1

    # loading model from path (runs/.../[<run_name>])
    cfg.model.load_from_files = True
    cfg.model.model_files = experiment_path / "model_files"


    run(cfg)




