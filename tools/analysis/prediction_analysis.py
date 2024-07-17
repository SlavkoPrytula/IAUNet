import os
from os import mkdir, makedirs
from os.path import join
import os.path as osp

import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import argparse
from tqdm import tqdm
from itertools import islice
import json

import sys
sys.path.append("./")

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

from visualizations.coco_vis import save_coco_vis


def run(cfg: _cfg):
    # set seed for reproducibility.
    set_seed(cfg.seed)
    

    # get dataloaders
    dataset = DATASETS.get(cfg.dataset.type)
    train_dataset = dataset(cfg, 
                            dataset_type="train", 
                            normalization=normalize, 
                            transform=valid_transforms(cfg)
                            )
    # valid_dataset = dataset(cfg, 
    #                         dataset_type="valid",
    #                         normalization=normalize, 
    #                         transform=valid_transforms(cfg)
    #                         )
    
    valid_dataset = dataset(cfg, 
                            dataset_type="valid",
                            normalization=normalize, 
                            transform=valid_transforms(cfg)
                            )
    
    train_dataloader = build_loader(train_dataset, batch_size=cfg.train.batch_size, num_workers=1)
    valid_dataloader = build_loader(valid_dataset, batch_size=cfg.valid.batch_size, num_workers=1)

    # build and prepare model.
    model = build_model(cfg)
    model.eval()

    # evaluate.
    # evaluator = EVALUATORS.get(cfg.model.evaluator.type)(cfg=cfg, model=model, dataset=valid_dataset)  # from config
    # evaluator(valid_dataloader)
    # evaluator.evaluate(verbose=True)


    for i, batch in enumerate(valid_dataloader):
        if batch is None:
            continue
        
        # prepare targets
        images = []
        targets = []

        target = batch[0]
        ignore = ["img_id", "img_path", "ori_shape", "file_name", "coco_id"]
        target = {k: v.to(cfg.device) if k not in ignore else v 
                for k, v in target.items()}
        images.append(target["image"])
        targets.append(target)

        image = nested_tensor_from_tensor_list(images)


        # ============= PREDICTION ==============
        # predict.
        pred = model(image.tensors)

        scores = pred['pred_logits'].softmax(-1)
        scores = scores[0, :, :-1].flatten(0, 1)
        iams_pred = pred["pred_iam"]
        iams_pred = iams_pred[0, ...]
        mask_feats = pred["mask_feats"]
        mask_feats = mask_feats[0, ...]

        mask_feats = torch.sum(mask_feats, dim=0).cpu().detach().numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(mask_feats, cmap='jet')
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig(f"./mask_feats_{i}.png", bbox_inches='tight', pad_inches=0)
        plt.close()



        # keep = scores > 0.3
        # iams_pred = iams_pred[keep]

        # N, H, W = iams_pred.shape
        # iams_pred = F.softmax(iams_pred.view(N, -1), dim=-1)
        # iams_pred = iams_pred.view(N, H, W)

        # # iams_pred = iams_pred.sigmoid()

        # combined_iam = torch.sum(iams_pred, dim=0)
        # combined_iam = (combined_iam - combined_iam.min()) / (combined_iam.max() - combined_iam.min())
        # combined_iam = combined_iam.cpu().detach().numpy()

        # print(combined_iam.max())
        # combined_iam[combined_iam > 0.15] = 0.05


        # plt.figure(figsize=(10, 10))
        # plt.imshow(combined_iam, cmap='jet')
        # plt.axis('off')
        # plt.tight_layout()
        
        # plt.savefig(f"./combined_iam_{i}.png", bbox_inches='tight', pad_inches=0)
        # plt.close()
    

    # # save results.
    # stats = evaluator.stats
    # results_dir = join(cfg.save_dir, 'results', 'evaluation_results.json')
    # with open(results_dir, 'w') as file:
    #     json.dump(stats, file, indent=4)
    

    # # plot results.
    # gt_coco = evaluator.gt_coco
    # pred_coco = evaluator.pred_coco

    # # TODO: 2config : Visualizations {n_samples: int = 5}
    # # n_samples = len(valid_dataset)
    # n_samples = 6
    # for batch in islice(valid_dataloader, n_samples):
    #     targets = batch[0]
        
    #     img = targets["image"][0]
    #     fname = targets["file_name"]
    #     idx = targets["coco_id"]
    #     H, W = targets["ori_shape"]
    #     out_file = join(cfg.visuals_dir, f'{fname}.jpg')

    #     save_coco_vis(img, gt_coco, pred_coco, idx, shape=[H, W], path=out_file)


def get_config_from_path(path: str) -> _cfg:
    try:
        from models import import_from_file
        module = import_from_file(join(path, "config_files/base.py"))
        config = getattr(module, 'cfg')
    except:
        config = _cfg
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation with IAUNet')
    parser.add_argument('--experiment_name', type=str, default='', help='name of the experiment')
    return parser.parse_args()


if __name__ == '__main__':
    import sys
    sys.path.append("./")
    args = parse_args()

    experiment_path = Path("runs/[resnet_iaunet_multitask]/[brightfield_coco_v2.0]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[base]/[job=51034854]-[2024-04-24 16:39:57]")
    cfg = get_config_from_path(experiment_path)
    old_dataset = cfg.dataset.name
    
    # cfg.dataset = "rectangle"
    # cfg.dataset = "worms"
    # cfg.dataset = "brightfield"
    # cfg.dataset = "brightfield_coco"
    cfg.dataset = "brightfield_coco_v2.0"
    # cfg.dataset = "EVICAN2Easy"
    # cfg.dataset = "EVICAN2Medium"
    # cfg.dataset = "EVICAN2Difficult"
    # cfg.dataset = "LiveCell"
    # cfg.dataset = "LiveCell2Percent"
    # cfg.dataset = "LiveCell30Images"
    # cfg.dataset = "YeastNet"
    # cfg.dataset = "HuBMAP"
    cfg.dataset = DATASETS_CFG.get(cfg.dataset)
    
    # model params
    cfg.model.evaluator=dict(
        type="MMDetDataloaderEvaluator",
        # type="AnalysisDataloaderEvaluator",
        mask_thr=0.5,
        score_thr=0.1,
        nms_thr=0.5,
        metric='segm', 
        classwise=True,
        outfile_prefix="results/coco"
    )

    cfg.model.criterion.matcher=dict(
        type='HungarianMatcher',
        cost_dice=5.0,
        cost_cls=2.0,
        cost_mask=5.0
    )


    cfg.run.run_name = join(cfg.run.run_name.replace(old_dataset, cfg.dataset.name))
    cfg.run.exist_ok = False
    
    # TODO: make load_pretrained unified from loading pretrained model and model from file (weights_path)
    cfg.model.weights = experiment_path / "checkpoints/best.pth" 
    cfg.model.load_pretrained = True
    cfg.model.save_model_files = False

    cfg.valid.batch_size = 1
    cfg.train.batch_size = 1
    cfg.train.n_folds = 5
    # cfg.valid.size = [768, 768]

    # loading model from path (runs/.../[<run_name>])
    cfg.model.load_from_files = True
    cfg.model.model_files = experiment_path / "model_files"

    run(cfg)

