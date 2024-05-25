import os
from os import mkdir, makedirs
from os.path import join
import gc
import importlib.util

import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
from utils.coco.coco import COCO

from utils.utils import nested_tensor_from_tensor_list
from utils.visualise import visualize_grid_v2

import argparse
from tqdm import tqdm


# @hydra.main(version_base=None, config_name="config")
def run(cfg: cfg):
    # create directories.
    cfg.save_dir = increment_path(
        join(cfg.run.runs_dir, "evals", cfg.run.experiment_name, cfg.run.run_name), 
        exist_ok=cfg.run.exist_ok
        )
    print(cfg.save_dir)
    cfg.visuals_dir = cfg.save_dir / 'visuals'
    makedirs(cfg.visuals_dir, exist_ok=True)

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

        # get dataloaders
        train_loader, valid_loader = get_dataloaders(cfg, df, fold=fold_i)

        # build and prepare model
        model = load_model(cfg, cfg.model.weights)
        # model.eval()

        # idx = 0
        # for idx in range(1, 5):
        #     # get predictions
        #     # TODO: prepare targets in dataloader - done
        #     for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        #         # prepare targets
        #         images = []
        #         targets = []
        #         for i in range(len(batch)):
        #             target = batch[i]

        #             target = {k: v.to(cfg.device) for k, v in target.items()}
        #             images.append(target["image"])

        #             targets.append(target)
                    
        #         images = nested_tensor_from_tensor_list(images)   # (B, C, H, W)
                
        #         output = model(images.tensors, idx)

        #         # if step == 3:
        #         break

        #     vis_preds_cyto = output['pred_masks'].sigmoid().cpu().detach().numpy()
        #     vis_logits_cyto  = output['pred_logits'].sigmoid().cpu().detach().numpy()
        #     # vis_preds_iams = output['pred_iam'].sigmoid().cpu().detach().numpy()
        #     iam = output['pred_iam']
        #     vis_preds_iams = output['pred_iam'].cpu().detach().numpy()
        #     # for iam in vis_preds_iams[0]:
        #         # print(iam.min(), iam.max())

        #     B, N, H, W = iam.shape
            
        #     visualize_grid_v2(
        #         masks=vis_preds_cyto[0, ...], 
        #         titles=vis_logits_cyto[0, :, 0],
        #         ncols=5, 
        #         path=f'{cfg.visuals_dir}/cyto_{i}.jpg'
        #     )
            
        #     # visualize_grid_v2(
        #     #     masks=vis_preds_iams[0, ...], 
        #     #     titles=vis_logits_cyto[0, :, 0],
        #     #     ncols=5, 
        #     #     path=f'{cfg.visuals_dir}/iam_[iam_init={idx}].jpg', 
        #     #     cmap='jet',
        #     #     # vmin=0, vmax=1
        #     # )


        #     # -----------
        #     # IAM Logits.  
        #     vis_preds_iams = iam.cpu().detach().numpy()
            
        #     visualize_grid_v2(
        #         masks=vis_preds_iams[0, ...], 
        #         titles=vis_logits_cyto[0, :, 0],
        #         ncols=5, 
        #         path=f'{cfg.visuals_dir}/iam_logits_{idx}.jpg',
        #         cmap='jet',
        #         # vmin=0, vmax=1
        #     )
            

        #     # -----------
        #     # IAM Sigmoid.
        #     vis_preds_iams = iam.sigmoid().cpu().detach().numpy()
            
        #     visualize_grid_v2(
        #         masks=vis_preds_iams[0, ...], 
        #         titles=vis_logits_cyto[0, :, 0],
        #         ncols=5, 
        #         path=f'{cfg.visuals_dir}/iam_sigmoid_{idx}.jpg',
        #         cmap='jet',
        #         # vmin=0, vmax=1
        #     )

            
        #     # -----------
        #     # IAM Softmax.  
        #     iam = F.softmax(iam.view(B, N, -1), dim=-1)
        #     iam = iam.view(B, N, H, W)
        #     vis_preds_iams = iam.cpu().detach().numpy()
            
        #     visualize_grid_v2(
        #         masks=vis_preds_iams[0, ...], 
        #         titles=vis_logits_cyto[0, :, 0],
        #         ncols=5, 
        #         path=f'{cfg.visuals_dir}/iam_softmax_{idx}.jpg',
        #         cmap='jet',
        #         vmin=0, vmax=1
        #     )

        #     torch.cuda.empty_cache()
        #     gc.collect()
            
        # raise

        # evaluate.
        evaluator = DataloaderEvaluator(cfg=cfg)
        evaluator(model, valid_loader)
        evaluator.evaluate()

        # plot results.
        gt_coco = COCO(evaluator.gt_coco)
        pred_coco = COCO(evaluator.pred_coco)

        for i in range(1, 3):
            img = np.zeros((512, 512))
            fig, ax = plt.subplots(1, 2, figsize=[20, 10])

            annIds  = gt_coco.getAnnIds(imgIds=[i])
            anns    = gt_coco.loadAnns(annIds)
            ax[0].imshow(img)
            gt_coco.showAnns(anns, draw_bbox=False, ax=ax[0])
            plt.tight_layout()

            annIds  = pred_coco.getAnnIds(imgIds=[i])
            anns    = pred_coco.loadAnns(annIds)
            ax[1].imshow(img)
            pred_coco.showAnns(anns, draw_bbox=False, ax=ax[1])
            plt.tight_layout()

            fig.savefig(join(cfg.visuals_dir, f'visual_{i}.jpg'))
            plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation with SparseUnet')
    parser.add_argument('--experiment_name', type=str, default='', help='name of the experiment')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[brightfield_nuc]-[sigmoid_iam]-[multi_level=True]-[coord_conv=False]-[losses=['labels', 'masks']]/[2023-07-08 12:37:28]")
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[rectangle]-[softmax_iam]-[multi_level=True]-[coord_conv=False]-[losses=['labels', 'masks', 'iam']]/[2023-07-07 19:56:09]")
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[brightfield_nuc]-[sigmoid_iam]-[multi_level=True]-[coord_conv=False]-[losses=['labels', 'masks']]/[2023-07-09 01:59:04]")
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[brightfield_nuc]-[sigmoid_iam]-[multi_level=True]-[coord_conv=False]-[losses=['labels', 'masks']]/[2023-07-09 09:39:28]")

    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[rectangle]-[sigmoid_iam]-[multi_level=True]-[coord_conv=False]-[losses=['labels', 'masks']]/[job=43858538]-[2023-07-10 01:04:24]")
    
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[rectangle]-[softmax_iam]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=43910856]-[2023-07-10 16:34:00]")
    
    # best rectangle.
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[rectangle]-[softmax_iam]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=44023978]-[2023-07-12 11:14:21]")
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[rectangle]-[softmax_iam]-[multi_level=True]-[coord_conv=False]-[losses=['labels', 'masks']]/[job=44039077]-[2023-07-12 15:17:40]")

    
    
    # no ovlp
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=45675080]-[2023-08-04 10:12:49]")
    
    # ovlp + inst | single iam
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'overlaps']]/[job=45677956]-[2023-08-04 12:12:01]")
    
    # ovlp, inst | group iam
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'overlaps']]/[job=45736477]-[2023-08-05 13:42:26]")
    
    # ovlp -> pob(ovlp), inst | group iam
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'overlaps']]/[job=45752670]-[2023-08-05 18:12:46]")
    
    # (ovlp * inst), inst | group iam
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'overlaps']]/[job=45753376]-[2023-08-05 18:21:54]")
    
    # cat(ovlp, inst) -> pob(ovlp), inst | group iam
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'overlaps']]/[job=45772926]-[2023-08-06 01:01:20]")


    # concat group iam
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'overlaps']]/[job=45803502]-[2023-08-06 16:47:02]")

    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'overlaps']]/[job=45820412]-[2023-08-06 21:19:18]")
    experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'overlaps']]/[job=45817975]-[2023-08-06 20:41:18]")

    config_path = experiment_path / "default.yaml"
    cfg.yaml_load(config_path)

    cfg.run.run_name = join(cfg.run.run_name, args.experiment_name)
    cfg.run.exist_ok = False

    # cfg.dataset.coco_dataset = join(cfg.project.home_dir, f'data/datasets/synthetic_datasets/rectangle/rectangles_[n=100]_[R_min=2_R_max=15]_[06.08.23].json')
    cfg.dataset.coco_dataset = join(cfg.project.home_dir, f'data/datasets/synthetic_datasets/rectangle/rectangles_[n=100]_[R_min=2_R_max=5]_[06.08.23].json')
    
    # TODO: make load_pretrained unified from loading pretrained model and model from file (weights_path)
    cfg.model.weights = experiment_path / "checkpoints/best.pth"
    cfg.model.load_pretrained = True
    cfg.model.save_model_files = False

    cfg.valid.bs = 1
    cfg.train.bs = 1
    cfg.train.n_folds = 1

    # loading model from path (runs/...)
    cfg.model.load_from_files = True
    cfg.model.model_files = experiment_path / "model_files"

    run(cfg)




