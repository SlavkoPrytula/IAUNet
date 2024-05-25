import os
from os import mkdir, makedirs
from os.path import join
import os.path as osp
import gc
import importlib.util

import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

from configs import cfg
from configs.base import dict

# from dataset.dataloaders import get_dataloaders
from models.build_model import build_model, load_model
from dataset.prepare_dataset import get_folds

# from dataset.datasets.brightfiled import df as _df
# from dataset.datasets.rectangle import df as _df

from utils.seed import set_seed
from utils.cuda import cuda_init

from configs.utils import save_config
from utils.files import increment_path

# from utils.evaluate.dataloader_evaluator import DataloaderEvaluator
from utils.evaluate import DataloaderEvaluator, MMDetDataloaderEvaluator
from utils.coco.coco import COCO

from utils.utils import nested_tensor_from_tensor_list, flatten_mask
from utils.visualise import visualize_grid_v2, visualize

import argparse
from tqdm import tqdm

from utils.augmentations import train_transforms, valid_transforms
from utils.normalize import normalize

from utils.optimizers import *
from utils.schedulers import *
from models.seg.loss import *
from models.seg.matcher import *

from dataset.dataloaders import build_loader
from utils.registry import DATASETS, EVALUATORS

from configs.datasets import DATASETS_CFG


def run(cfg: cfg):
    # create directories.
    cfg.save_dir = increment_path(
        join(cfg.run.runs_dir, "evals", cfg.run.experiment_name, cfg.run.run_name), 
        exist_ok=cfg.run.exist_ok
        )
    print(cfg.save_dir)

    cfg.visuals_dir = cfg.save_dir / 'visuals'
    makedirs(cfg.visuals_dir, exist_ok=True)
    makedirs(cfg.save_dir / 'results', exist_ok=True)

    # set seed for reproducibility
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
                            dataset_type="eval",
                            normalization=normalize, 
                            transform=valid_transforms(cfg)
                            )
    
    train_dataloader = build_loader(train_dataset, batch_size=cfg.train.batch_size, num_workers=2)
    valid_dataloader = build_loader(valid_dataset, batch_size=cfg.valid.batch_size, num_workers=2)

    # build and prepare model
    model = build_model(cfg)
    model.eval()

    # idx = 0
    # for idx in range(1, 10):
        # get predictions
        # TODO: prepare targets in dataloader - done
    # for idx, batch in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
    #     # prepare targets
    #     images = []
    #     targets = []
    #     for i in range(len(batch)):
    #         target = batch[i]

    #         target = {k: v.to(cfg.device) for k, v in target.items()}
    #         images.append(target["image"])

    #         targets.append(target)
            
    #     images = nested_tensor_from_tensor_list(images)   # (B, C, H, W)
        
    #     output = model(images.tensors, idx)

    # #     if idx == 14:
    # #         break

    #     vis_preds_cyto = output['pred_masks'].sigmoid().cpu().detach().numpy()
    #     vis_logits_cyto  = output['pred_logits'].sigmoid().cpu().detach().numpy()
    #     # vis_preds_iams = output['pred_iam'].sigmoid().cpu().detach().numpy()
    # #     iam = output['pred_iam']
    # #     vis_preds_iams = output['pred_iam'].cpu().detach().numpy()
    # #     # for iam in vis_preds_iams[0]:
    # #         # print(iam.min(), iam.max())


    #     # vis_preds_occl = output[f'pred_occluders'].sigmoid().cpu().detach().numpy()

    #     # visualize_grid_v2(
    #     #     masks=vis_preds_occl[0, ...], 
    #     #     titles=vis_logits_cyto[0, :, 0],
    #     #     ncols=5, 
    #     #     path=f'{cfg.visuals_dir}/occluders_{idx}.jpg'
    #     # )

    # #     B, N, H, W = iam.shape
        
    #     visualize_grid_v2(
    #         masks=vis_preds_cyto[0, ...], 
    #         titles=vis_logits_cyto[0, :, 0],
    #         ncols=5, 
    #         path=f'{cfg.visuals_dir}/cyto_{idx}.jpg'
    #     )

    #     visualize_grid_v2(
    #         masks=vis_preds_cyto[0, ...], 
    #         titles=vis_logits_cyto[0, :, 0],
    #         ncols=10, 
    #         path=f'{cfg.visuals_dir}/big_cyto_{idx}.jpg'
    #     )
    #     break
    
        # visualize_grid_v2(
        #     masks=vis_preds_iams[0, ...], 
        #     titles=vis_logits_cyto[0, :, 0],
        #     ncols=5, 
        #     path=f'{cfg.visuals_dir}/iam_[iam_init={idx}].jpg', 
        #     cmap='jet',
        #     # vmin=0, vmax=1
        # )

        # TODO: Class for plotting
        # -----------
        # IAM Logits.  
        # vis_preds_iams = iam.cpu().detach().numpy()
        
        # visualize_grid_v2(
        #     masks=vis_preds_iams[0, ...], 
        #     titles=vis_logits_cyto[0, :, 0],
        #     ncols=5, 
        #     path=f'{cfg.visuals_dir}/iam_logits_{idx}.jpg',
        #     cmap='jet',
        #     # vmin=0, vmax=1
        # )
        

        # -----------
        # IAM Sigmoid.
        # vis_preds_iams = iam.sigmoid().cpu().detach().numpy()
        # vis_preds_iams = flatten_mask(iam.sigmoid().cpu().detach().numpy()[0, ...], axis=0)[0, ...]
        # visualize(
        #     images=vis_preds_iams,
        #     cmap='jet',
        #     path=f'{cfg.visuals_dir}/iam_sigmoid_{idx}.jpg'
        # )
        
        # visualize_grid_v2(
        #     masks=vis_preds_iams[0, ...], 
        #     titles=vis_logits_cyto[0, :, 0],
        #     ncols=5, 
        #     path=f'{cfg.visuals_dir}/iam_sigmoid_{idx}.jpg',
        #     cmap='jet',
        #     # vmin=0, vmax=1
        # )

        
        # -----------
        # IAM Softmax.  
        # iam = F.softmax(iam.view(B, N, -1), dim=-1)
        # iam = iam.view(B, N, H, W)
        # # vis_preds_iams = iam.cpu().detach().numpy()

        # vis_preds_iams = flatten_mask(iam.cpu().detach().numpy()[0, ...], axis=0)[0, ...]
        # visualize(
        #     images=vis_preds_iams,
        #     cmap='jet',
        #     path=f'{cfg.visuals_dir}/iam_softmax_{idx}.jpg'
        # )
        
        # visualize_grid_v2(
        #     masks=vis_preds_iams[0, ...], 
        #     titles=vis_logits_cyto[0, :, 0],
        #     ncols=5, 
        #     path=f'{cfg.visuals_dir}/iam_softmax_{idx}.jpg',
        #     cmap='jet',
        #     # vmin=0, vmax=1
        # )

        # torch.cuda.empty_cache()
        # gc.collect()
        
    # raise
    print(len(valid_dataset))

    # evaluate.
    # evaluator = DataloaderEvaluator(cfg=cfg.model.evaluator)  # explicit
    # evaluator = MMDetDataloaderEvaluator(cfg=cfg, dataset=valid_dataset)
    evaluator = EVALUATORS.get(cfg.model.evaluator.type)(cfg=cfg, dataset=train_dataset)  # from config
    evaluator(model, train_dataloader)
    evaluator.evaluate(verbose=True)

    # plot results.
    # gt_coco = COCO(evaluator.gt_coco)
    # pred_coco = COCO(evaluator.pred_coco)
    gt_coco = evaluator.gt_coco
    pred_coco = evaluator.pred_coco

    for i in range(1, 5):
        targets = train_dataset[i-1]
        idx = valid_dataset.image_ids[i-1]
        img_path = targets["img_path"]
        fname, name = osp.splitext(osp.basename(img_path))
        out_file = join(cfg.visuals_dir, f'{fname}.jpg')

        H, W = targets["ori_shape"]
        img = targets["image"][0]

        fig, ax = plt.subplots(1, 2, figsize=[20, 10])
        fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        # img1
        annIds  = gt_coco.getAnnIds(imgIds=[idx])
        anns    = gt_coco.loadAnns(annIds)
        ax[0].imshow(img, cmap='gray')

        gt_masks = gt_coco.getMasks(anns, alpha=0.3)
        for gt_mask in gt_masks:
            gt_mask = cv2.resize(gt_mask, (W, H))
            ax[0].imshow(gt_mask)

        # img2
        annIds  = pred_coco.getAnnIds(imgIds=[idx])
        anns    = pred_coco.loadAnns(annIds)
        ax[1].imshow(img, cmap='gray')

        pred_masks = gt_coco.getMasks(anns, alpha=0.3)
        for pred_mask in pred_masks:
            pred_mask = cv2.resize(pred_mask, (W, H))
            ax[1].imshow(pred_mask)


        for a in ax:
            a.axis('off')
            a.set_xlim(0, W)
            a.set_ylim(H, 0)

        fig.canvas.draw()
        fig.savefig(out_file, bbox_inches='tight', pad_inches=0)
        plt.close(fig)



    # for i in range(1, 7):
    #     targets = valid_dataset[i-1]
    #     img_path = targets["img_path"]
    #     fname, name = osp.splitext(osp.basename(img_path))
    #     out_file = join(cfg.visuals_dir, f'{fname}.jpg')

    #     H, W = valid_dataset[i-1]["ori_shape"]
    #     img = targets["image"][0]

    #     total_width = W * 2
    #     dpi = 100
    #     fig, ax = plt.subplots(1, 2, figsize=[20, 10])
    #     fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    #     # img1
    #     annIds  = gt_coco.getAnnIds(imgIds=[i])
    #     anns    = gt_coco.loadAnns(annIds)
    #     ax[0].imshow(img, cmap='gray')

    #     gt_masks = gt_coco.getMasks(anns, alpha=0.5)
    #     for gt_mask in gt_masks:
    #         gt_mask = cv2.resize(gt_mask, (W, H))
    #         ax[0].imshow(gt_mask)

    #     # img2
    #     annIds  = pred_coco.getAnnIds(imgIds=[i])
    #     anns    = pred_coco.loadAnns(annIds)
    #     ax[1].imshow(img, cmap='gray')

    #     pred_masks = gt_coco.getMasks(anns, alpha=0.5)
    #     for pred_mask in pred_masks:
    #         pred_mask = cv2.resize(pred_mask, (W, H))
    #         ax[1].imshow(pred_mask)


    #     for a in ax:
    #         a.axis('off')
    #         a.set_xlim(0, W)
    #         a.set_ylim(H, 0)

    #     fig.canvas.draw()
    #     fig.savefig(out_file, bbox_inches='tight', pad_inches=0)
    #     plt.close(fig)



def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation with SparseUnet')
    parser.add_argument('--experiment_name', type=str, default='', help='name of the experiment')
    return parser.parse_args()


if __name__ == '__main__':
    import sys
    sys.path.append("./")
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
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'overlaps']]/[job=45817975]-[2023-08-06 20:41:18]")
    
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=45867499]-[2023-08-07 11:44:24]")

    # experiments = [
    #     # Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'overlaps']]/[job=45820412]-[2023-08-06 21:19:18]")
        
    #     # Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'overlaps']]/[job=45977581]-[2023-08-08 20:59:37]"),
    #     Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=46036037]-[2023-08-09 13:25:51]"),
    #     Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'overlaps']]/[job=46034550]-[2023-08-09 13:00:46]"),
    # ]
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=46036037]-[2023-08-09 13:25:51]")

    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=45966022]-[2023-08-08 17:15:00]")
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'occluders']]/[job=46576294]-[2023-08-16 13:27:56]")
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'occluders']]/[job=46483565]-[2023-08-15 11:27:50]")
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_occluder]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'occluders']]/[job=46591201]-[2023-08-16 16:47:17]")
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_occluder]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'occluders']]/[job=46611364]-[2023-08-16 22:51:59]")
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_occluder]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'occluders']]/[job=46632905]-[2023-08-17 11:18:59]")

    # base model [masks] - best
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=45966022]-[2023-08-08 17:15:00]")
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'occluders']]/[job=46483565]-[2023-08-15 11:27:50]")
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_occluder]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'occluders']]/[job=46611364]-[2023-08-16 22:51:59]")

    # ======================= Exhaustive =========================
    # base model [masks] - new - good results ~0.7 (no nms)
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[sigmoid_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=48862059]-[2023-10-28 16:41:51]")
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[sigmoid_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=48862145]-[2023-10-29 01:13:59]")

    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[sigmoid_iam]/[kernel_dim=128]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=48912077]-[2023-10-31 18:04:14]")
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[sigmoid_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49001319]-[2023-11-02 09:50:32]")
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[sigmoid_iam]/[kernel_dim=512]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49001329]-[2023-11-02 09:54:18]")

    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49155908]-[2023-11-11 16:02:18]")


    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[softmax_iam]/[kernel_dim=128]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49173438]-[2023-11-12 00:06:53]")
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49173439]-[2023-11-12 00:12:53]")
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[softmax_iam]/[kernel_dim=512]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49173440]-[2023-11-12 00:18:53]")

    # ablation studies | kernel size ✅ 
    # new - [ms]
    # kernel_size=128
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[softmax_iam]/[kernel_dim=128]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49177327]-[2023-11-12 13:49:27]")
    # kernel_size=256
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49177916]-[2023-11-12 14:19:32]")
    # kernel_size=512
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[softmax_iam]/[kernel_dim=512]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49177398]-[2023-11-12 13:53:19]")

    # ablation studies | coord conv ✅ 
    # no coord_conv
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=False]-[losses=['labels', 'masks']]/[job=49178216]-[2023-11-12 15:57:38]")

    # ablation studies | model size ✅
    # 3-level + 2-layers + 2-iam layers (iaunet_s)
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49179479]-[2023-11-13 03:39:37]")
    # 4-level + 4-layers + 4-iam layers (ia_unet_m)
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49179480]-[2023-11-13 03:41:37]")
    # 5-level + 4-layers + 4-iam layers (ia_unet_l)
    # - same as [kernel_size=256]

    # ablation studies | activation functions ✅
    # softmax
    # - same as [kernel_size=256]
    # temp_softmax
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[temp_softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49179477]-[2023-11-13 03:33:38]")
    # sigmoid
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[sigmoid_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49179772]-[2023-11-13 09:53:49]")

    # ablation studies | bipartite matching costs ✅
    # cls + dice
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49186502]-[2023-11-13 19:08:15]")
    # cls + dice + bce
    # - same as [kernel_size=256]


    # experiment_path = Path("runs/[sparse_seunet]/[YeastNet]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49173531]-[2023-11-12 02:52:56]")



    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[softmax_iam]/[kernel_dim=256]-[multi_level=False]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49181016]-[2023-11-13 13:31:56]")
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[sigmoid_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=48862144]-[2023-10-29 01:10:11]")
    # runs/[sparse_seunet]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=45966022]-[2023-08-08 17:15:00]



    # ========================= EVICAN2 =========================
    # evican2
    # experiment_path = Path("runs/[sparse_seunet]/[EVICAN2]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49088920]-[2023-11-08 10:53:55]")


    # ======================== LiveCell =========================
    # livecell
    # experiment_path = Path("runs/[sparse_seunet]/[LiveCell30Images]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49109061]-[2023-11-08 21:51:03]")
    # experiment_path = Path("runs/[sparse_seunet]/[LiveCell2Percent]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49179205]-[2023-11-12 23:37:39]")


    # ======================== HuBMAP =========================
    # hubmap
    # experiment_path = Path("runs/[sparse_seunet]/[HuBMAP]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49189226]-[2023-11-14 02:54:30]")
    # experiment_path = Path("runs/[sparse_seunet]/[HuBMAP]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49188829]-[2023-11-14 02:38:30]")


    # ======================== YeastNet =========================
    # yeastnet
    # experiment_path = Path("runs/[sparse_seunet]/[YeastNet]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49173531]-[2023-11-12 02:52:56]")


    # ===========
    # fixed classification
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49269820]-[2023-11-23 08:16:36]")
    # experiment_path = Path("runs/[sparse_seunet]/[brightfield_coco]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49338211]-[2023-11-28 08:31:02]")
    experiment_path = Path("runs/[sparse_seunet]/[brightfield_coco_v2.0]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=50263497]-[2024-02-02 11:34:13]")
    





    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[synthetic_brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=46860262]-[2023-08-21 15:48:36]")
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=47119523]-[2023-08-26 20:54:15]")
    
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[occluders-sparse_seunet_occluder]-[512]/[brightfield]/[softmax_iam]/[multi_level=False]-[coord_conv=True]-[losses=['labels', 'masks', 'occluders']]/[job=48336077]-[2023-09-24 15:09:48]")
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=47322537]-[2023-08-30 14:16:24]")
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=47344335]-[2023-08-30 18:08:43]")
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[original_plus_synthetic_brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=47429309]-[2023-08-31 23:10:46]")


    # datasets = [
    #     'data/datasets/synthetic_datasets/rectangle/rectangles_[n=100]_[R_min=2_R_max=15]_[06.08.23].json',
    #     # 'data/datasets/synthetic_datasets/rectangle/rectangles_[n=100]_[R_min=2_R_max=5]_[06.08.23].json'
    # ]
    
    # for experiment_path in experiments:
    #     print(experiment_path)

    #     for dataset in datasets:
    #         print(dataset)

    # config_path = experiment_path / "default.yaml"
    # cfg.yaml_load(config_path)

    # cfg = cfg.get_config_from_path(experiment_path)
    # print(cfg)

    old_dataset = cfg.dataset.name
    
    # cfg.dataset = "brightfield"
    # cfg.dataset = "brightfield_coco"
    cfg.dataset = "brightfield_coco_v2.0"
    # cfg.dataset = "EVICAN2Easy"
    # cfg.dataset = "EVICAN2Medium"
    # cfg.dataset = "EVICAN2Difficult"
    # cfg.dataset = "LiveCell"
    # cfg.dataset = "LiveCell30Images"
    # cfg.dataset = "YeastNet"
    # cfg.dataset = "HuBMAP"
    cfg.dataset = DATASETS_CFG.get(cfg.dataset)
    
    # model params
    # cfg.model.in_channels = 1
    cfg.model.num_masks = 100
    cfg.model.kernel_dim = 256
    cfg.model.mask_dim = 256

    cfg.model.evaluator=dict(
        type="MMDetDataloaderEvaluator",
        # type="AnalysisDataloaderEvaluator",
        mask_thr=0.1,
        cls_thr=-1,
        score_thr=0.05,
        nms_thr=0.5,
    )

    cfg.model.criterion.matcher=dict(
        type='HungarianMatcher',
        cost_dice=5.0,
        cost_cls=2.0,
        cost_mask=5.0
    )


    cfg.run.run_name = join(cfg.run.run_name.replace(old_dataset, cfg.dataset.name), args.experiment_name)
    cfg.run.exist_ok = False

    # cfg.dataset.coco_dataset = join(cfg.project.home_dir, dataset)
    # cfg.dataset.coco_dataset = join(cfg.project.home_dir, f'data/datasets/synthetic_datasets/rectangle/rectangles_[n=100]_[R_min=2_R_max=15]_[06.08.23].json')
    # cfg.dataset.coco_dataset = join(cfg.project.home_dir, f'data/datasets/synthetic_datasets/rectangle/rectangles_[n=100]_[R_min=2_R_max=5]_[06.08.23].json')
    
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

    run(cfg)




# eval_results = inference_on_dataset(
#     model,
#     data_loader,
#     DatasetEvaluators([COCOEvaluator(...), Counter()]))


# TODO: register datasets and create custom mappers
# - so for each evaluation i can set multiple dataset mappers for the same dataset to test



# datasets = ["dataset_name_0", "dataset_name_1", ...]
# models = [Path(0), Path(1), ....]

# register_dataset("dataset_name", nn.Dataset)
# train_loader = DatasetMapper("dataset_name", "train")
# valid_loader = DatasetMapper("dataset_name", "valid")

# model = build_model(cfg)
# results = inference_on_dataset(
#     model, 
#     valid_loader, 
#     Evaluators([DataloaderEvaluator(...)]))

