import os
from os import mkdir, makedirs
from os.path import join
import gc
import importlib.util
import re

import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from configs import cfg

from models.build_model import build_model, load_model
from dataset.prepare_dataset import get_folds

from utils.seed import set_seed
from configs.utils import save_config
from utils.files import increment_path

from utils.evaluate.dataloader_evaluator import DataloaderEvaluator
from utils.coco.coco import COCO

from utils.utils import nested_tensor_from_tensor_list, flatten_mask
from utils.visualise import visualize_grid_v2, visualize

import argparse
from tqdm import tqdm

from utils.augmentations import train_transforms, valid_transforms
from utils.normalize import normalize


from utils.seed import set_seed

from utils.optimizers import *
from utils.schedulers import *
from models.seg.loss import *
from models.seg.matcher import *

from dataset.dataloaders import build_loader
from utils.registry import DATASETS


def inference_on_dataset(model, data_loader, evaluator: DataloaderEvaluator):
    evaluator(model, data_loader)
    evaluator.evaluate(verbose=False)
    results = evaluator.stats

    return results


def run(cfg: cfg, model):
    # create directories.
    # cfg.save_dir = increment_path(
    #     join(cfg.run.runs_dir, "evals", cfg.run.experiment_name, cfg.run.run_name), 
    #     exist_ok=cfg.run.exist_ok
    #     )

    # cfg.visuals_dir = cfg.save_dir / 'visuals'
    # makedirs(cfg.visuals_dir, exist_ok=True)

    # set seed for reproducibility
    set_seed(cfg.seed)


    evaluator = DataloaderEvaluator(cfg=cfg)
    
    # get dataloaders
    # train_loader, valid_loader = get_dataloaders(cfg, df, fold=0)
    dataset = DATASETS.get(cfg.dataset.name)
    train_dataset = dataset(cfg, 
                            is_train=True, 
                            normalization=normalize, 
                            transform=train_transforms(cfg)
                            )
    valid_dataset = dataset(cfg, 
                            is_train=False,
                            normalization=normalize, 
                            transform=valid_transforms(cfg)
                            )
    train_dataloader = build_loader(train_dataset, batch_size=cfg.train.batch_size, num_workers=2)
    valid_dataloader = build_loader(valid_dataset, batch_size=cfg.valid.batch_size, num_workers=2)

    print()
    print()
    job = re.search(r'\[job=(\d+)\]', cfg.run.run_name).group(1)
    print(f'Running experiment: {job}')
    # print(f'Running experiment: {cfg.run.experiment_name}/{cfg.run.run_name}')
    print(f'Evaluating on dataset: {cfg.dataset.name}')

    # evaluate.
    results = inference_on_dataset(model, valid_dataloader, evaluator)
    
    print()

    metrics = list(results.keys())
    vals = list(results.values())

    msg = ('%13s,' * len(metrics) % tuple(metrics)).replace(',', '') + '\n'
    msg += ('%13.5g,' * len(vals) % tuple(vals)).replace(',', '')
    print(msg)

    print()
    print()


def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation with SparseUnet')
    parser.add_argument('--experiment_name', type=str, default='', help='name of the experiment')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    experiments = [
        # Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'overlaps']]/[job=45820412]-[2023-08-06 21:19:18]")
        
        # Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'overlaps']]/[job=45977581]-[2023-08-08 20:59:37]"),
                
        #  - best
        # Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=46036037]-[2023-08-09 13:25:51]"),
        # Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'overlaps']]/[job=46034550]-[2023-08-09 13:00:46]"),

        # Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'overlaps']]/[job=46293818]-[2023-08-12 17:11:52]"),
        # Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'overlaps']]/[job=46309545]-[2023-08-12 21:32:03]"),
        # Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'overlaps']]/[job=46329611]-[2023-08-13 11:59:17]"),

        # Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'occluders']]/[job=46329639]-[2023-08-13 12:41:52]"),
        # Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'occluders']]/[job=46329802]-[2023-08-13 16:27:53]"),
        # Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'occluders']]/[job=46352428]-[2023-08-14 00:05:58]"),

        # Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_occluder]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'occluders']]/[job=46354562]-[2023-08-14 00:40:50]"),
        
        # Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'occluders']]/[job=46397052]-[2023-08-14 12:15:06]"),

        # Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'occluders']]/[job=46431691]-[2023-08-14 22:02:16]"),
        # Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'occluders']]/[job=46432240]-[2023-08-14 22:11:09]"),
        
        # Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_occluder_ml]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'occluders']]/[job=46802775]-[2023-08-20 17:40:24]"),
        # Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_occluder_gcn_mh]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'occluders']]/[job=46787404]-[2023-08-20 11:38:00]"),
        # Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_occluder_gcn_mh]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'occluders']]/[job=46735648]-[2023-08-19 12:03:34]"),
        # Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_occluder_gcn_mh]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'occluders']]/[job=46752648]-[2023-08-19 19:22:06]"),


        # Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=47423656]-[2023-08-31 20:48:47]"),
        # Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=47423974]-[2023-08-31 20:52:41]"),
        Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[original_plus_synthetic_brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=47429309]-[2023-08-31 23:10:46]"),
        Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[original_plus_synthetic_brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=47430513]-[2023-08-31 23:51:12]"),

    ]


    datasets = [
        # 'data/datasets/synthetic_datasets/rectangle/rectangles_[n=100]_[R_min=2_R_max=15]_[06.08.23].json',
        # 'data/datasets/synthetic_datasets/rectangle/rectangles_[n=100]_[R_min=2_R_max=5]_[06.08.23].json'
        
        # 'data/datasets/synthetic_datasets/rectangle/rectangles_[S_min=20_S_max=100]_[n=1000]_[R_min=5_R_max=20]_[overlap=0.5]_[15.08.23].json',
        # 'data/datasets/synthetic_datasets/rectangle/rectangles_[S_min=100_S_max=300]_[n=1000]_[R_min=5_R_max=10]_[overlap=0.5]_[15.08.23].json',
        # 'data/datasets/synthetic_datasets/rectangle/rectangles_[S_min=100_S_max=300]_[n=1000]_[R_min=2_R_max=3]_[overlap=0.5]_[15.08.23].json',

    ]
    
    for experiment_path in experiments:

        # setup.
        config_path = experiment_path / "default.yaml"
        cfg.yaml_load(config_path)
        
        cfg.run.run_name = join(cfg.run.run_name, args.experiment_name)
        cfg.run.exist_ok = False
        
        cfg.model.weights = experiment_path / "checkpoints/best.pth"
        cfg.model.load_pretrained = True
        cfg.model.save_model_files = False

        cfg.valid.batch_size = 1
        cfg.train.batch_size = 1
        cfg.train.n_folds = 5

        # loading model from path (runs/.../[<run_name>])
        cfg.model.load_from_files = True
        cfg.model.model_files = experiment_path / "model_files"

        # build and prepare model
        model = build_model(cfg)
        model.eval()


        # for dataset in datasets:
        #     cfg.dataset.coco_dataset = join(cfg.project.home_dir, dataset)

        run(cfg, model)
            
        print("=" * 100)




# eval_results = inference_on_dataset(
#     model,
#     data_loader,
#     DatasetEvaluators([COCOEvaluator(...), Counter()]))


# TODO: register datasets and create custom mappers
# - so for each evaluation i can set multiple dataset mappers for the same dataset to test



# datasets = ["dataset_name_0", "dataset_name_1", ...]
# models = [Path(0), Path(1), ....]

# register_dataset("dataset_name", nn.Dataset)
# train_loader = DatasetMapper("dataset_name", "train")
# valid_loader = DatasetMapper("dataset_name", "valid")

# model = build_model(cfg)
# results = inference_on_dataset(
#     model, 
#     valid_loader, 
#     Evaluators([DataloaderEvaluator(...)]))



#               cfg
#                |
# DatasetMapper("dataset_name", "valid") -> factory_mapper