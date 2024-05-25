import os
from os import mkdir, makedirs
from os.path import join

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from configs import cfg
from dataset.dataloaders import get_dataloaders
from models.build_model import build_model, load_model
from dataset.prepare_dataset import get_folds

from dataset.datasets.brightfiled import df as _df
# from dataset.datasets.rectangle import df as _df

from utils.seed import set_seed
from utils.cuda import cuda_init

from configs.utils import save_config
from utils.files import increment_path

from utils.evaluate.dataloader_evaluator import DataloaderEvaluator
from utils.coco.coco import COCO


from tqdm import tqdm
from utils.utils import nested_tensor_from_tensor_list
from utils.visualise import visualize_grid_v2

# @hydra.main(version_base=None, config_name="config")
def run(cfg: cfg):
    # create directories.
    cfg.save_dir = increment_path(join(cfg.run.runs_dir, "evals", cfg.run.experiment_name, cfg.run.run_name), exist_ok=cfg.run.exist_ok)
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
        model = load_model(cfg, cfg.weights_path)

        for idx in range(5):
            # get predictions
            # TODO: prepare targets in dataloader
            for step, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                images = []
                targets = []
                for i in range(len(batch)):
                    bf_images, cyto_masks, masks_iam = batch[i]
                    bf_images = bf_images.to(cfg.device, dtype=torch.float)
                    masks_cyto = cyto_masks.to(cfg.device)
                    masks_iam = masks_iam.to(cfg.device)
                    
                    # image
                    images.append(bf_images)
                    
                    # labels
                    N, _, _ = masks_cyto.shape
                    gt_labels = torch.zeros(N, dtype=torch.int64)
                    gt_labels = gt_labels.to(cfg.device)

                    # targets
                    target = {
                        'labels': gt_labels,
                        'masks': masks_cyto,
                    }
                    targets.append(target)
                    
                images = nested_tensor_from_tensor_list(images)   # (B, C, H, W)
                
                output = model(images.tensors)

                break

            vis_preds_cyto = output['pred_masks'].sigmoid().cpu().detach().numpy()
            vis_logits_cyto  = output['pred_logits'].sigmoid().cpu().detach().numpy()
            vis_preds_iams = output['pred_iam'].sigmoid().cpu().detach().numpy()
            
            visualize_grid_v2(
                masks=vis_preds_cyto[0, ...], 
                titles=vis_logits_cyto[0, :, 0],
                ncols=10, 
                path=f'{cfg.visuals_dir}/cyto.jpg'
            )
            
            
            visualize_grid_v2(
                masks=vis_preds_iams[0, ...], 
                titles=vis_logits_cyto[0, :, 0],
                ncols=10, 
                path=f'{cfg.visuals_dir}/iam.jpg', 
                cmap='jet',
                # vmin=0, vmax=1
            )
            
        raise

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


if __name__ == '__main__':
    experiment_path = Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[brightfield_nuc]-[sigmoid_iam]-[multi_level=True]-[coord_conv=False]-[losses=['labels', 'masks']]/[2023-07-08 12:37:28]")
    config_path = experiment_path / "default.yaml"
    cfg.yaml_load(config_path)

    # cfg.weights_path = "/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[rectangle]-[prompt_iam]-[softmax_iam]2/checkpoints/best.pth"
    # cfg.weights_path = "/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[rectangle]-[prompt_iam]-[sigmoid_iam]-[coord_conv]/checkpoints/best.pth"

    cfg.weights_path = experiment_path / "checkpoints/best.pth"
    cfg.run.exist_ok = False

    run(cfg)

