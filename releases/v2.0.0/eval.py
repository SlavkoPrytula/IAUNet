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
# from dataset.dataloaders import get_dataloaders
from models.build_model import build_model, load_model
from dataset.prepare_dataset import get_folds

# from dataset.datasets.brightfiled import df as _df
# from dataset.datasets.rectangle import df as _df

from utils.seed import set_seed
from utils.cuda import cuda_init

from configs.utils import save_config
from utils.files import increment_path

from utils.evaluate.dataloader_evaluator import DataloaderEvaluator
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
    

    # Run training
    for fold_i in [0]:
        print(f'+ Fold: {fold_i}')
        print(f'-' * 10)
        print()

        # get dataloaders
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

        # build and prepare model
        # model = load_model(cfg, cfg.model.weights)
        model = build_model(cfg)
        model.eval()

        # idx = 0
        # for idx in range(1, 10):
            # get predictions
            # TODO: prepare targets in dataloader - done
        for idx, batch in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader)):
            # prepare targets
            images = []
            targets = []
            for i in range(len(batch)):
                target = batch[i]

                target = {k: v.to(cfg.device) for k, v in target.items()}
                images.append(target["image"])

                targets.append(target)
                
            images = nested_tensor_from_tensor_list(images)   # (B, C, H, W)
            
            output = model(images.tensors, idx)

        #     if idx == 14:
        #         break

            vis_preds_cyto = output['pred_masks'].sigmoid().cpu().detach().numpy()
            vis_logits_cyto  = output['pred_logits'].sigmoid().cpu().detach().numpy()
            # vis_preds_iams = output['pred_iam'].sigmoid().cpu().detach().numpy()
        #     iam = output['pred_iam']
        #     vis_preds_iams = output['pred_iam'].cpu().detach().numpy()
        #     # for iam in vis_preds_iams[0]:
        #         # print(iam.min(), iam.max())


            # vis_preds_occl = output[f'pred_occluders'].sigmoid().cpu().detach().numpy()

            # visualize_grid_v2(
            #     masks=vis_preds_occl[0, ...], 
            #     titles=vis_logits_cyto[0, :, 0],
            #     ncols=5, 
            #     path=f'{cfg.visuals_dir}/occluders_{idx}.jpg'
            # )

        #     B, N, H, W = iam.shape
            
            visualize_grid_v2(
                masks=vis_preds_cyto[0, ...], 
                titles=vis_logits_cyto[0, :, 0],
                ncols=5, 
                path=f'{cfg.visuals_dir}/cyto_{idx}.jpg'
            )

            visualize_grid_v2(
                masks=vis_preds_cyto[0, ...], 
                titles=vis_logits_cyto[0, :, 0],
                ncols=10, 
                path=f'{cfg.visuals_dir}/big_cyto_{idx}.jpg'
            )
            break
        
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

        # evaluate.
        # evaluator = DataloaderEvaluator(cfg=cfg.model.evaluator)  # explicit
        evaluator = EVALUATORS.get(cfg.model.evaluator.type)(cfg=cfg.model.evaluator)  # from config
        evaluator(model, valid_dataloader)
        evaluator.evaluate(verbose=True)

        # plot results.
        gt_coco = COCO(evaluator.gt_coco)
        pred_coco = COCO(evaluator.pred_coco)

        for i in range(1, 7):
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


    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[synthetic_brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=46860262]-[2023-08-21 15:48:36]")
    # experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=47119523]-[2023-08-26 20:54:15]")
    
    experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[hornet]-[512]/[brightfield]/[softmax_iam]/[multi_level=False]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=47819353]-[2023-09-06 14:18:50]")
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

    config_path = experiment_path / "default.yaml"
    cfg.yaml_load(config_path)

    cfg.run.run_name = join(cfg.run.run_name, args.experiment_name)
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

