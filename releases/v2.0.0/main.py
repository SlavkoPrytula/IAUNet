import os
from os import mkdir, makedirs
from os.path import join
import torch

# logging
# import hydra
# from hydra.core.config_store import ConfigStore

from configs import cfg
from models.build_model import build_model
from engine.run_training import run_training

from dataset.dataloaders import build_loader
from utils.augmentations import train_transforms, valid_transforms
from utils.normalize import normalize


from utils.seed import set_seed
from configs.utils import save_config
from utils.files import increment_path

from utils.optimizers import *
from utils.schedulers import *
from utils.evaluate import *
from models.seg.loss import *
from models.seg.matcher import *

from utils.registry import build_from_cfg, build_criterion, build_matcher, build_optimizer, build_scheduler
from utils.registry import DATASETS, OPTIMIZERS, SCHEDULERS, CRITERIONS, EVALUATORS


cfg.save_dir = increment_path(join(cfg.run.runs_dir, cfg.run.experiment_name, cfg.run.run_name), exist_ok=cfg.run.exist_ok)

# save config.
print(cfg)
# save_config(cfg, cfg.save_dir)
# from pathlib import Path
# save_config(cfg, Path("./"))

# save visuals.
makedirs(cfg.save_dir / 'train_visuals', exist_ok=True)
makedirs(cfg.save_dir / 'valid_visuals', exist_ok=True)
makedirs(cfg.save_dir / 'checkpoints', exist_ok=True)

# save results.
cfg.csv = cfg.save_dir / 'results.csv'

print(cfg.__dict__())

# set logger.
# cfg.log = cfg.save_dir / 'output.log'
# set_logging(name=LOGGING_NAME, log_file=cfg.log, verbose=True)  # run before defining LOGGER


# @hydra.main(version_base=None, config_name="config")
def run(cfg: cfg):
    # set seed for reproducibility
    set_seed(cfg.seed)

    # - get dataloaders
    # train_loader, valid_loader = get_dataloaders(cfg, df, fold=fold_i)
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

    # - build and prepare model
    model = build_model(cfg)
    print(model)

    cfg.optimizer.params = model.parameters()
    # optimizer = OPTIMIZERS.build(cfg.optimizer)
    optimizer = build_optimizer(cfg.optimizer)


    cfg.scheduler.optimizer = optimizer
    # scheduler = SCHEDULERS.build(cfg.scheduler)
    scheduler = build_scheduler(cfg.scheduler)


    # loss = CRITERIONS.build(cfg.model.criterion)
    cfg.model.criterion.save_dir = cfg.save_dir
    criterion = build_criterion(cfg.model.criterion)
    
    evaluator = EVALUATORS.get(cfg.model.evaluator.type)(cfg=cfg.model.evaluator)
    
    
    # - run training
    model = run_training(cfg, model, 
                         criterion=criterion, 
                         train_dataloader=train_dataloader, 
                         valid_dataloader=valid_dataloader,
                         optimizer=optimizer, 
                         scheduler=scheduler,
                         evaluator=evaluator
                         )




if __name__ == '__main__':
    run(cfg)



# Examples:
# # loss = CRITERIONS.build(cfg.model.criterion)
# loss = build_criterion(cfg.model.criterion)
# print(loss)

# # matcher = MATCHERS.build(cfg.model.criterion.matcher)
# matcher = build_matcher(cfg.model.criterion.matcher, MATCHERS)
# print(matcher)