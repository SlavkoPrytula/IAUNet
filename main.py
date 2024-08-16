
from os import makedirs
from os.path import join
from pathlib import Path
from datetime import datetime
import wandb

import hydra
from omegaconf import OmegaConf
from configs import cfg, experiment_name

from dataset.dataloaders import (build_loader, 
                                 build_loader_ms,
                                 empty_collate_fn, 
                                 metadata_collate_fn, 
                                 trivial_batch_collator, 
                                 worker_init_fn)
from utils.augmentations import train_transforms, valid_transforms

from utils.seed import set_seed
from utils.dist.comm import setup, cleanup
from utils.logging import setup_logger

import torch.multiprocessing as mp

from utils.optimizers import *
from utils.schedulers import *
from utils.callbacks import *
from evaluation import *
from models.seg.loss import *
from models.seg.matcher import *

from utils.registry import build_from_cfg, build_criterion, build_matcher, build_optimizer, build_scheduler
from utils.registry import DATASETS, OPTIMIZERS, SCHEDULERS, CRITERIONS, EVALUATORS, CALLBACKS

from models.build_model import build_model
from engine.trainer import Trainer

TIME = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def run(rank: int = 0, world_size: int = 1, cfg: cfg = None):
    distributed = cfg.trainer.get('strategy') == 'ddp'
    setup(rank, world_size, distributed=distributed)

    if cfg.job_id and cfg.run_id:
        cfg.run.save_dir = join(cfg.run.save_dir, f"[job={cfg.job_id}]-[group_run]", f"run={cfg.run_id}")
    elif cfg.job_id and (not cfg.run_id):
        cfg.run.save_dir = join(cfg.run.save_dir, f"[job={cfg.job_id}]-[{TIME}]")
    else:
        print("\nNo JOB_ID found for this run! Using default 'temp' folder.\n")
        cfg.run.save_dir = join(cfg.run.save_dir, "temp")

    cfg.run.save_dir = Path(cfg.run.save_dir)
    print(f"Saving to: {cfg.run.save_dir}\n")

    # create directories.
    makedirs(cfg.run.save_dir, exist_ok=True)
    makedirs(cfg.run.save_dir / 'train_visuals', exist_ok=True)
    makedirs(cfg.run.save_dir / 'checkpoints', exist_ok=True)
    makedirs(cfg.run.save_dir / 'results', exist_ok=True)
    makedirs(cfg.run.save_dir / 'config_files', exist_ok=True)

    # save config.
    print(OmegaConf.to_yaml(cfg))
    config_path = cfg.run.save_dir / "config_files" / "train.yaml"
    OmegaConf.save(config=cfg, f=config_path)

    # set logger.
    logger = setup_logger(
        name=cfg.logger.log.name, 
        log_files=cfg.logger.log.log_files
        )

    # wandb.
    # wandb.init(
    #     project=cfg.logger.wandb.project, 
    #     group=cfg.logger.wandb.group,
    #     name=cfg.logger.wandb.name,
    #     dir=cfg.run.save_dir
    #     )

    # ============================================================
    # ============================================================

    set_seed(cfg.seed)

    # - get dataloaders
    dataset = DATASETS.get(cfg.dataset.type)

    train_dataset = dataset(cfg, 
                            dataset_type="train", 
                            transform=train_transforms(cfg)
                            )
    valid_dataset = dataset(cfg, 
                            dataset_type="valid",
                            transform=valid_transforms(cfg)
                            )
    # eval_dataset = dataset(cfg, 
    #                         dataset_type="eval",
    #                         transform=valid_transforms(cfg)
    #                         )

    train_dataloader = build_loader(train_dataset, 
                                    batch_size=cfg.dataset.train_dataset.batch_size, 
                                    num_workers=8, #cfg.trainer.num_workers, 
                                    collate_fn=trivial_batch_collator, 
                                    seed=cfg.seed, 
                                    distributed=distributed)
    valid_dataloader = build_loader(valid_dataset, 
                                    batch_size=cfg.dataset.valid_dataset.batch_size, 
                                    num_workers=8, #cfg.trainer.num_workers, 
                                    collate_fn=trivial_batch_collator, 
                                    seed=cfg.seed, 
                                    distributed=distributed)
    # eval_dataloader = build_loader(eval_dataset, 
    #                                 batch_size=cfg.dataset.eval_dataset.batch_size, 
    #                                 num_workers=8, #cfg.trainer.num_workers, 
    #                                 collate_fn=trivial_batch_collator, 
    #                                 seed=cfg.seed, 
    #                                 distributed=distributed)
    
    # - build and prepare model
    model = build_model(cfg)

    optimizer_cfg = OmegaConf.to_container(cfg.model.optimizer, resolve=True)
    optimizer_cfg['params'] = model.parameters()
    optimizer = OPTIMIZERS.build(optimizer_cfg)

    scheduler_cfg = OmegaConf.to_container(cfg.model.scheduler, resolve=True)
    scheduler_cfg['optimizer'] = optimizer
    scheduler = SCHEDULERS.build(scheduler_cfg)

    criterion = CRITERIONS.build(cfg.model.criterion)

    
    # TODO: this needs to be refactored
    # the evaluation should have information about the dataset
    # evaluator = EVALUATORS.get(cfg.model.evaluator.type)(cfg=cfg.model.evaluator)
    # potentially with this you can add multiple datasets.
    from evaluation import OverlapIOUEvaluator
    evaluators = {
        "valid_coco": EVALUATORS.get(cfg.model.evaluator.type)(cfg=cfg, model=model, dataset=valid_dataset),
        "overlap_iou": OverlapIOUEvaluator(cfg=cfg, model=model, dataset=valid_dataset)
    }

    # setup callbacks.
    callbacks = [CALLBACKS.build(cfg.callbacks[c]) for c in cfg.callbacks]


    # - run training.
    trainer = Trainer(cfg, model, 
                      criterion=criterion, 
                      train_dataloader=train_dataloader, 
                      valid_dataloader=valid_dataloader,
                      optimizer=optimizer, 
                      scheduler=scheduler,
                      evaluators=evaluators,
                      callbacks=callbacks,
                      logger=logger,
                      rank=rank,
                      strategy=cfg.trainer.get('strategy'),
                      sync_batchnorm=cfg.trainer.get('sync_batchnorm')
                    )
    trainer.train()

    if cfg.test:
        # TODO: run testing on model
        UserWarning("Testing not implemented! Check main.py")

    wandb.finish()

    if distributed:
        cleanup()


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: cfg):
    if cfg.trainer.get('strategy') == 'ddp':
        world_size = cfg.trainer.devices
        mp.spawn(run, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        run(cfg=cfg)


if __name__ == '__main__':
    main()


# Examples:
# # loss = CRITERIONS.build(cfg.model.criterion)
# loss = build_criterion(cfg.model.criterion)
# print(loss)

# # matcher = MATCHERS.build(cfg.model.criterion.matcher)
# matcher = build_matcher(cfg.model.criterion.matcher, MATCHERS)
# print(matcher)