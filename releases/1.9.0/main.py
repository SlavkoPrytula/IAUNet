import os
from os import mkdir, makedirs
from os.path import join

import pandas as pd
import torch
from IPython.core.display import display

import logging.config

# logging
# import hydra
# from hydra.core.config_store import ConfigStore

from configs import cfg, LOGGING_NAME, set_logging
from dataset.dataloaders import get_dataloaders
from models.build_model import build_model
from engine.run_training import run_training
# from utils.optimizers import SWA
from utils.schedulers import fetch_scheduler
from dataset.prepare_dataset import get_folds

from dataset.datasets.brightfiled import df as _df
# from dataset.datasets.synthetic_brightfield import df as _df
# from dataset.datasets.rectangle import df as _df

from utils.seed import set_seed
from utils.cuda import cuda_init

from configs.utils import save_config
from utils.files import increment_path


cfg.save_dir = increment_path(join(cfg.run.runs_dir, cfg.run.experiment_name, cfg.run.run_name), exist_ok=cfg.run.exist_ok)

# save config.
print(cfg)
save_config(cfg, cfg.save_dir)

# save visuals.
makedirs(cfg.save_dir / 'train_visuals', exist_ok=True)
makedirs(cfg.save_dir / 'valid_visuals', exist_ok=True)
makedirs(cfg.save_dir / 'checkpoints', exist_ok=True)

# save results.
cfg.csv = cfg.save_dir / 'results.csv'

# set logger.
# cfg.log = cfg.save_dir / 'output.log'
# set_logging(name=LOGGING_NAME, log_file=cfg.log, verbose=True)  # run before defining LOGGER


# @hydra.main(version_base=None, config_name="config")
def run(cfg: cfg):
    # select gpu device
    # cuda_init(cfg.gpus)
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

        # - get dataloaders
        train_loader, valid_loader = get_dataloaders(cfg, df, fold=fold_i)

        # - build and prepare model
        model = build_model(cfg)
        
        losses = {}
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
        scheduler = fetch_scheduler(cfg, optimizer)

        # - run training
        model = run_training(cfg, model, losses, optimizer, scheduler=scheduler,
                             train_loader=train_loader, valid_loader=valid_loader,
                             num_epochs=cfg.train.epochs, run=None, fold_i=fold_i)





if __name__ == '__main__':
    run(cfg)