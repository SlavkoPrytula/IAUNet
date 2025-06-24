from os import makedirs
from os.path import join
from pathlib import Path
from datetime import datetime
import wandb

import hydra
from omegaconf import OmegaConf
from configs import cfg

from dataset.dataloaders import (build_loader, trivial_batch_collator)
from utils.augmentations import train_transforms, valid_transforms

from utils.optimizers import *
from utils.schedulers import *
from utils.callbacks import *
from evaluation import *
from models import *

from utils.registry import build_from_cfg, build_criterion, build_matcher, build_optimizer, build_scheduler
from utils.registry import DATASETS, OPTIMIZERS, SCHEDULERS, CRITERIONS, EVALUATORS, CALLBACKS, MODELS

from models.factory import build_model
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from utils.logging.lightning_logger import PLLogger

TIME = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def build_optimizer(cfg, model):
    optimizer_cfg = cfg['optimizer']
    base_lr = optimizer_cfg.get("lr", 1e-4)
    base_weight_decay = optimizer_cfg.get("weight_decay", 0.05)

    backbone_lr_multiplier = cfg.get("backbone_lr_multiplier", 0.1)
    embedding_weight_decay = cfg.get("embedding_weight_decay", 0.0)

    params = []
    memo = set()

    backbone_names = ["backbone", "encoder"]
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue  # Skip frozen parameters
            if param in memo:
                continue  # Avoid duplicates
            memo.add(param)

            hyperparams = {
                "lr": base_lr,
                "weight_decay": base_weight_decay,
            }

            if any(name in module_name.lower() for name in backbone_names):
                hyperparams["lr"] *= backbone_lr_multiplier
                print(f"Setting LR for {module_name} to {hyperparams['lr']}")
            if isinstance(module, torch.nn.Embedding):
                hyperparams["weight_decay"] = embedding_weight_decay
                print(f"Setting weight decay for {module_name} to {hyperparams['weight_decay']}")

            params.append({"params": param, **hyperparams})

    optimizer_cfg["params"] = params
    return optimizer_cfg


def run(cfg: cfg):
    # ============================================================
    pl.seed_everything(cfg.seed, workers=True)

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
    cfg.logger.log.log_files = [str(cfg.run.save_dir / log) for log in cfg.logger.log.log_files]
    logger = PLLogger(
        name="iaunet", 
        log_files=cfg.logger.log.log_files,
        save_dir=cfg.run.save_dir,
        )

    # wandb.
    if cfg.logger.get('wandb'):
        wandb.init(
            project='IAUNet', 
            group=cfg.logger.wandb.group,
            name=f'job_id={cfg.job_id}',
            dir=cfg.run.save_dir
            )

    # ============================================================
    # ============================================================

    # - get dataloaders
    dataset = DATASETS.get(cfg.dataset.type)

    train_dataset = dataset(cfg, 
                            dataset_type="train", 
                            transform=train_transforms(cfg))
    valid_dataset = dataset(cfg, 
                            dataset_type="valid",
                            transform=valid_transforms(cfg))
    eval_dataset = dataset(cfg, 
                            dataset_type="eval",
                            transform=valid_transforms(cfg))

    train_dataloader = build_loader(train_dataset, 
                                    batch_size=cfg.dataset.train_dataset.batch_size, 
                                    num_workers=4, #cfg.trainer.num_workers, 
                                    collate_fn=trivial_batch_collator, 
                                    seed=cfg.seed, 
                                    shuffle=True)
    valid_dataloader = build_loader(valid_dataset, 
                                    batch_size=cfg.dataset.valid_dataset.batch_size, 
                                    num_workers=4, #cfg.trainer.num_workers, 
                                    collate_fn=trivial_batch_collator, 
                                    seed=cfg.seed, 
                                    shuffle=False)
    eval_dataloader = build_loader(eval_dataset, 
                                    batch_size=cfg.dataset.eval_dataset.batch_size, 
                                    num_workers=4, #cfg.trainer.num_workers, 
                                    collate_fn=trivial_batch_collator, 
                                    seed=cfg.seed, 
                                    shuffle=False)
    
    print(MODELS)
    # - build and prepare model
    model = build_model(cfg)
    # model = IAUNet(cfg)
    print(model)

    print(train_dataset[0].labels)
    raise
    
    evaluators = {
        "valid": {
            "coco": MMDetDataloaderEvaluator(cfg=cfg, dataset=valid_dataset), 
        },
        "eval": {
            "coco": MMDetDataloaderEvaluator(cfg=cfg, dataset=eval_dataset), 
        },
    }

    callbacks = {c: CALLBACKS.build(cfg.callbacks[c]) for c in cfg.callbacks}
    csv_logger_callback = CSVLogger(save_dir=cfg.run.save_dir)
    callbacks['csv_logger'] = csv_logger_callback

    coco_eval_callback = CocoEval(cfg, save_coco_vis=False)
    coco_eval_callback.evaluators = evaluators
    callbacks['coco_eval'] = coco_eval_callback
    
    print(f"Using callbacks: {list(callbacks.keys())}")
    callbacks = list(callbacks.values())

    # - trainer.
    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.strategy,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        max_epochs=cfg.trainer.max_epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        deterministic=cfg.trainer.deterministic,
        benchmark=cfg.trainer.benchmark,
        precision=cfg.trainer.precision,
        sync_batchnorm=cfg.trainer.sync_batchnorm,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_model_summary=cfg.trainer.enable_model_summary,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
        profiler=cfg.trainer.profiler,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        gradient_clip_algorithm=cfg.trainer.gradient_clip_algorithm
    )

    trainer.fit(
        model, 
        train_dataloaders=train_dataloader, 
        val_dataloaders=valid_dataloader,
        ckpt_path=cfg.model.ckpt_path
    )
    
    if cfg.test:
        trainer.test(model, dataloaders=eval_dataloader)


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: cfg):
    run(cfg=cfg)


if __name__ == '__main__':
    main()
