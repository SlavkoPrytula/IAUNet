import os
from os import mkdir, makedirs
from os.path import join
import torch
import argparse
from pathlib import Path
from datetime import datetime
import wandb

# logging
# import hydra
# from hydra.core.config_store import ConfigStore

from dataset.dataloaders import (build_loader, 
                                 build_loader_ms,
                                 empty_collate_fn, 
                                 metadata_collate_fn, 
                                 trivial_batch_collator, 
                                 worker_init_fn)
from utils.augmentations import train_transforms, valid_transforms
from utils.augmentations import normalize


from utils.seed import set_seed
from configs.utils import save_config
from utils.files import increment_path
from utils.comm import setup, cleanup
from utils.logging import setup_logger

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.optimizers import *
from utils.schedulers import *
from utils.evaluate import *
from models.seg.loss import *
from models.seg.matcher import *
from visualizations.visualizers import *

from utils.registry import build_from_cfg, build_criterion, build_matcher, build_optimizer, build_scheduler
from utils.registry import DATASETS, OPTIMIZERS, SCHEDULERS, CRITERIONS, EVALUATORS, VISUALIZERS

from configs import cfg, LOGGING_NAME
from models.build_model import build_model
from engine.run_training import run_training

TIME = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def parse_args():
    parser = argparse.ArgumentParser(description='Train job')
    parser.add_argument('--job-id', default='', help='Job ID for this run')
    parser.add_argument('--run-id', default='', help='Subdirectory for this run')
    args = parser.parse_args()

    return args


args = parse_args()

# cfg.save_dir = increment_path(join(cfg.run.runs_dir, cfg.run.experiment_name, cfg.run.run_name), exist_ok=cfg.run.exist_ok)
cfg.save_dir = join(cfg.run.runs_dir, cfg.run.experiment_name, cfg.run.run_name, cfg.run.group_name)

if args.job_id and args.run_id:
    cfg.save_dir = join(cfg.save_dir, f"[job={args.job_id}]-[group_run]", f"run={args.run_id}")
elif args.job_id and (not args.run_id):
    cfg.save_dir = join(cfg.save_dir, f"[job={args.job_id}]-[{TIME}]")
else:
    print("\nNo JOB_ID found for this run! Using default 'temp' folder.\n")
    cfg.save_dir = join(cfg.save_dir, "temp")

cfg.save_dir = Path(cfg.save_dir)
print(f"Saving to {cfg.save_dir}\n")


# save config.
print(cfg)

# create directories.
makedirs(cfg.save_dir, exist_ok=True)
makedirs(cfg.save_dir / 'train_visuals', exist_ok=True)
# makedirs(cfg.save_dir / 'valid_visuals', exist_ok=True)
makedirs(cfg.save_dir / 'checkpoints', exist_ok=True)
makedirs(cfg.save_dir / 'results', exist_ok=True)

# save results.
cfg.csv = cfg.save_dir / 'results.csv'

# set logger.
# cfg.log = cfg.save_dir / 'output.log'
logger = setup_logger(name=LOGGING_NAME, log_files=cfg.logger.log_files)


# wandb.
# wandb.init(
#     project=cfg.wandb.project, 
#     group=cfg.wandb.group,
#     name=cfg.wandb.name,
#     dir=cfg.save_dir
#     )


# @hydra.main(version_base=None, config_name="config")
def run(cfg: cfg):
# def run(rank, world_size):
    # setup(rank, world_size)
    # set seed for reproducibility
    set_seed(cfg.seed)

    # - get dataloaders
    print(DATASETS)
    dataset = DATASETS.get(cfg.dataset.type)

    train_dataset = dataset(cfg, 
                            dataset_type="train", 
                            normalization=normalize, 
                            transform=train_transforms(cfg)
                            )
    valid_dataset = dataset(cfg, 
                            dataset_type="valid",
                            normalization=normalize, 
                            transform=valid_transforms(cfg)
                            )

    train_dataloader = build_loader(train_dataset, 
                                    batch_size=cfg.train.batch_size, 
                                    num_workers=4, 
                                    collate_fn=trivial_batch_collator, 
                                    seed=cfg.seed)
    valid_dataloader = build_loader(valid_dataset, 
                                    batch_size=cfg.valid.batch_size, 
                                    num_workers=4, 
                                    collate_fn=trivial_batch_collator, 
                                    seed=cfg.seed)
    
    # - build and prepare model
    model = build_model(cfg)

    cfg.optimizer.params = model.parameters()
    # optimizer = OPTIMIZERS.build(cfg.optimizer)
    optimizer = build_optimizer(cfg.optimizer)


    cfg.scheduler.optimizer = optimizer
    # scheduler = SCHEDULERS.build(cfg.scheduler)
    scheduler = build_scheduler(cfg.scheduler)


    # loss = CRITERIONS.build(cfg.model.criterion)
    cfg.model.criterion.save_dir = cfg.save_dir
    criterion = build_criterion(cfg.model.criterion)
    
    # TODO: this needs to be refactored
    # the evaluation should have information about the dataset
    # evaluator = EVALUATORS.get(cfg.model.evaluator.type)(cfg=cfg.model.evaluator)
    # potentially with this you can add multiple datasets.
    evaluators = {
        "valid": EVALUATORS.get(cfg.model.evaluator.type)(cfg=cfg, model=model, dataset=valid_dataset)
    }

    # if getattr(cfg.dataset, "occ_dataset", None):
    #     occ_dataset = dataset(cfg, 
    #                       dataset_type="occ", 
    #                       normalization=normalize, 
    #                       transform=valid_transforms(cfg)
    #                      )
        
    #     cfg.model.evaluator.coco_api = "COCOeval_nofp"
    #     evaluators["occ"] = EVALUATORS.get(cfg.model.evaluator.type)(cfg=cfg, model=model, dataset=occ_dataset)
    
    
    # - run training
    model = run_training(cfg, model, 
                         criterion=criterion, 
                         train_dataloader=train_dataloader, 
                         valid_dataloader=valid_dataloader,
                         optimizer=optimizer, 
                         scheduler=scheduler,
                         evaluators=evaluators,
                         device=cfg.device, 
                         logger=logger
                         )
    # cleanup()
    wandb.finish()




if __name__ == '__main__':
    run(cfg)

# if __name__ == "__main__":
#     world_size = cfg.gpus
#     mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)


# Examples:
# # loss = CRITERIONS.build(cfg.model.criterion)
# loss = build_criterion(cfg.model.criterion)
# print(loss)

# # matcher = MATCHERS.build(cfg.model.criterion.matcher)
# matcher = build_matcher(cfg.model.criterion.matcher, MATCHERS)
# print(matcher)