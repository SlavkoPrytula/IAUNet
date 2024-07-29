from os import mkdir, makedirs
from os.path import join
from pathlib import Path
import argparse

import sys
sys.path.append(".")

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


def evaluate(cfg: _cfg):
    # set seed for reproducibility.
    set_seed(cfg.seed)

    # cfg.save_dir = join(cfg.run.runs_dir, cfg.run.experiment_name, cfg.run.run_name)

    # get dataloaders
    dataset = DATASETS.get(cfg.dataset.type)
    valid_dataset = dataset(cfg, 
                            dataset_type="eval",
                            normalization=normalize, 
                            transform=valid_transforms(cfg)
                            )
    
    valid_dataloader = build_loader(valid_dataset, batch_size=cfg.valid.batch_size, num_workers=1)

    # build and prepare model.
    model = build_model(cfg)
    model.eval()

    # evaluate.
    evaluator = EVALUATORS.get(cfg.model.evaluator.type)(cfg=cfg, model=model, dataset=valid_dataset)  # from config
    evaluator(valid_dataloader)
    evaluator.evaluate(verbose=True)
    stats = evaluator.stats

    return stats



def run(paths):
    evaluation_results = []

    for experiment_path in paths:
        cfg = get_config_from_path(experiment_path)
        old_dataset = cfg.dataset.name
        
        # cfg.dataset = "brightfield"
        # cfg.dataset = "brightfield_coco"
        # cfg.dataset = "brightfield_coco_v2.0"
        # cfg.dataset = "EVICAN2Easy"
        # cfg.dataset = "EVICAN2Medium"
        # cfg.dataset = "EVICAN2Difficult"
        cfg.dataset = "LiveCell"
        # cfg.dataset = "LiveCell30Images"
        # cfg.dataset = "YeastNet"
        # cfg.dataset = "HuBMAP"
        cfg.dataset = DATASETS_CFG.get(cfg.dataset)
        
        # model params
        cfg.model.evaluator=dict(
            type="MMDetDataloaderEvaluator",
            mask_thr=0.5,
            score_thr=0.05,
            nms_thr=0.5,
            metric='segm', 
            classwise=True,
            # outfile_prefix="results/coco"
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

        results = evaluate(cfg)
        evaluation_results.append(results)


    sum_metrics = {key: 0 for key in evaluation_results[0].keys()}

    for result in evaluation_results:
        for key in sum_metrics:
            sum_metrics[key] += result[key]

    average_metrics = {key: value / len(evaluation_results) for key, value in sum_metrics.items()}

    for key, value in average_metrics.items():
        print(f"{key}: {value:.3f}")


if __name__ == '__main__':
    args = parse_args()

    experiment_paths = [
        Path("runs/[iaunet]/[LiveCell]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=50401340]-[group_run]/run=1"),
        Path("runs/[iaunet]/[LiveCell]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=50401340]-[group_run]/run=2"),
        Path("runs/[iaunet]/[LiveCell]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=50401340]-[group_run]/run=3"),
        Path("runs/[iaunet]/[LiveCell]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=50401340]-[group_run]/run=4"),
        Path("runs/[iaunet]/[LiveCell]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=50401340]-[group_run]/run=5"),
        ]
    
    run(experiment_paths)
