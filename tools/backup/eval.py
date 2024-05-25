import os
from os import mkdir, makedirs
from os.path import join

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

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


cfg.save_dir = increment_path(join(cfg.run.runs_dir, "evals", cfg.run.experiment_name, cfg.run.run_name), exist_ok=cfg.run.exist_ok)

cfg.visuals_dir = cfg.save_dir / 'visuals'
makedirs(cfg.visuals_dir, exist_ok=True)


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
        model = load_model(cfg, cfg.weights_path)
        
        # evaluate.
        evaluator = DataloaderEvaluator(cfg=cfg)
        evaluator(model, valid_loader)
        evaluator.evaluate()

        # plot results.
        gt_coco = COCO(evaluator.gt_coco)
        pred_coco = COCO(evaluator.pred_coco)

        img = np.zeros((512, 512))
        fig, ax = plt.subplots(1, 2, figsize=[20, 10])

        annIds  = gt_coco.getAnnIds(imgIds=[1])
        anns    = gt_coco.loadAnns(annIds)
        ax[0].imshow(img)
        gt_coco.showAnns(anns, draw_bbox=False, ax=ax[0])
        plt.tight_layout()

        annIds  = pred_coco.getAnnIds(imgIds=[1])
        anns    = pred_coco.loadAnns(annIds)
        ax[1].imshow(img)
        pred_coco.showAnns(anns, draw_bbox=False, ax=ax[1])
        plt.tight_layout()

        fig.savefig(cfg.visuals_dir)
        plt.close(fig)


if __name__ == '__main__':
    cfg.weights = "/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[rectangle]-[prompt_iam]-[softmax_iam]2/checkpoints/best.pth"
    run(cfg)

