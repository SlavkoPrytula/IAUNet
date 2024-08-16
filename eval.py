import os
from os import makedirs
from os.path import join

from pathlib import Path
import argparse
from itertools import islice
import json

import hydra
from omegaconf import OmegaConf
from configs import cfg, experiment_name

from configs import cfg as _cfg
from models.build_model import build_model
from utils.seed import set_seed
from utils.logging import setup_logger
from utils.augmentations import train_transforms, valid_transforms

from evaluation import *
from utils.optimizers import *
from utils.schedulers import *
from models.seg.loss import *
from models.seg.matcher import *

from dataset.dataloaders import build_loader, trivial_batch_collator
from utils.registry import DATASETS, EVALUATORS

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
    cfg.run.save_dir = Path(cfg.run.save_dir)
    print(f"Saving to {cfg.run.save_dir}\n")

    run_cfg = OmegaConf.create({
        'run': {
            'visuals_dir': cfg.run.save_dir / 'visuals',
            'results_dir': cfg.run.save_dir / 'results'
        }
    })
    cfg = OmegaConf.merge(cfg, run_cfg)
    
    makedirs(cfg.run.save_dir, exist_ok=True)
    makedirs(cfg.run.visuals_dir, exist_ok=True)
    makedirs(cfg.run.results_dir, exist_ok=True)

    # set logger.
    logger = setup_logger(
        name=cfg.logger.log.name, 
        log_files=cfg.logger.log.log_files
        )

    # set seed for reproducibility.
    set_seed(cfg.seed)

    # get dataloaders
    dataset = DATASETS.get(cfg.dataset.type)
    train_dataset = dataset(cfg, 
                            dataset_type="train", 
                            # normalization=normalize, 
                            transform=valid_transforms(cfg)
                            )
    
    valid_dataset = dataset(cfg, 
                            dataset_type="valid",
                            # normalization=normalize, 
                            transform=valid_transforms(cfg),
                            )
    
    train_dataloader = build_loader(train_dataset, 
                                    batch_size=cfg.dataset.train_dataset.batch_size, 
                                    num_workers=cfg.trainer.num_workers, 
                                    collate_fn=trivial_batch_collator, 
                                    seed=cfg.seed)
    valid_dataloader = build_loader(valid_dataset, 
                                    batch_size=cfg.dataset.valid_dataset.batch_size, 
                                    num_workers=cfg.trainer.num_workers, 
                                    collate_fn=trivial_batch_collator, 
                                    seed=cfg.seed)

    # build and prepare model.
    model = build_model(cfg)
    model.eval()

    if cfg.trainer.accelerator == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)

    # evaluate.
    evaluator = EVALUATORS.get(cfg.model.evaluator.type)(cfg=cfg, model=model, dataset=valid_dataset)  # from config
    evaluator(valid_dataloader)
    evaluator.evaluate(verbose=True)
    
    # save results.
    stats = evaluator.stats
    results_file = cfg.run.results_dir / 'evaluation_results.json'
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
        out_file = join(cfg.run.visuals_dir, f'{fname}.jpg')

        save_coco_vis(img, gt_coco, pred_coco, idx, shape=[H, W], path=out_file)


def get_config_from_path(path: str) -> _cfg:
    try:
        config_path = join(path, "config_files", "train.yaml")
        config = OmegaConf.load(config_path)
    except Exception as e:
        print(f"Error loading configuration from {path}: {e}")
        config = _cfg  # fallback to default
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation with IAUNet')
    parser.add_argument('--experiment_name', type=str, default='', help='name of the experiment')
    return parser.parse_args()


if __name__ == '__main__':
    import sys
    sys.path.append("./")
    args = parse_args()

    experiment_path = Path("runs/[iaunet]/[iadecoder]/[ResNet]/[worms]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[InstanceHead-v2.0-attn]/[job=51901826]-[2024-08-15 02:03:01]")
    cfg = get_config_from_path(experiment_path)
    old_dataset = cfg.dataset.name
    
    # cfg.dataset = "rectangle"
    cfg.dataset.name = "worms"
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

    eval_cfg = OmegaConf.create({
        'model': {
            'evaluator': {
                'type': "MMDetDataloaderEvaluator",
                'mask_thr': 0.5,
                'score_thr': 0.1,
                'nms_thr': 0.5,
                'metric': 'segm',
                'classwise': True,
                'outfile_prefix': "results/coco",
            },
            'criterion': {
                'matcher': {
                    'type': 'HungarianMatcher',
                    'cost_dice': 2.0,
                    'cost_cls': 2.0,
                    'cost_mask': 5.0,
                }
            },
            'weights': experiment_path / "checkpoints/best.pth",
            'load_pretrained': True,
            'save_model_files': False,
            'load_from_files': True,
            'model_files': experiment_path / "model_files",
        },
        'run': {
            'run_name': join(cfg.run.run_name.replace(old_dataset, cfg.dataset.name)),
            'save_dir': join(cfg.run.runs_dir, "evals", cfg.run.experiment_name, 
                             cfg.run.run_name, cfg.run.group_name, str(experiment_path).split("/")[-1]),
        },
        'dataset': {
            'valid_dataset': {
                'batch_size': 16
            },
            'eval_dataset': {
                'batch_size': 16,
            }
        }
    })
    cfg = OmegaConf.merge(cfg, eval_cfg)

    trainer_cfg = OmegaConf.create({
        "trainer": {
            "accelerator": "gpu",
            "devices": 1,
            "num_workers": 2,
            "strategy": None,
        }
    })
    cfg: _cfg = OmegaConf.merge(cfg, trainer_cfg)

    run(cfg)



# python eval.py