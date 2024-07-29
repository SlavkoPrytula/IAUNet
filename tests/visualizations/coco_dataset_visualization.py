import os
from os import mkdir, makedirs
from os.path import join
import os.path as osp

import sys
sys.path.append("./")

from configs import cfg

from utils.augmentations import train_transforms, valid_transforms
from utils.normalize import normalize

from utils.evaluate import *
from utils.optimizers import *
from utils.schedulers import *
from models.seg.loss import *
from models.seg.matcher import *

from dataset.dataloaders import build_loader
from utils.registry import DATASETS, EVALUATORS
from configs.datasets import DATASETS_CFG

from visualizations.coco_vis import save_coco_vis



def main(cfg: cfg):
    cfg.save_dir = "./tests/visualizations/results"
    makedirs(cfg.save_dir, exist_ok=True)

    # get dataloaders
    dataset = DATASETS.get(cfg.dataset.type)
    valid_dataset = dataset(cfg, 
                            dataset_type="valid",
                            normalization=normalize, 
                            transform=valid_transforms(cfg)
                            )
    
    print(len(valid_dataset))
    
    valid_dataloader = build_loader(valid_dataset, batch_size=cfg.valid.batch_size, num_workers=2)

    # evaluate.
    evaluator = EVALUATORS.get(cfg.model.evaluator.type)(cfg=cfg, dataset=valid_dataset)  # from config
    evaluator.gt_coco = evaluator.coco_metric._coco_api
    evaluator.pred_coco = evaluator.coco_metric._coco_api

    # plot results.
    gt_coco = evaluator.gt_coco
    pred_coco = evaluator.pred_coco
    

    for i in range(1):
        targets = next(iter(valid_dataloader))[0]

        img = targets["image"][0]
        fname = targets["file_name"]
        idx = targets["coco_id"]
        H, W = targets["ori_shape"]
        out_file = join(cfg.save_dir, f'{fname}.jpg')

        save_coco_vis(img, gt_coco, pred_coco, idx, shape=[H, W], path=out_file)

        # --------------------------------------------------------------- #
        # this code has been encapsulated in the save_coco_vis() function #
        # --------------------------------------------------------------- #
        # fig, ax = plt.subplots(1, 2, figsize=[20, 10])
        # fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        # # img1
        # ax[0].imshow(img, cmap='gray')
        # visualize_coco_anns(gt_coco, idx, ax[0], shape=[W, H], alpha=0.75, draw_border=True)

        # # img2
        # ax[1].imshow(img, cmap='gray')
        # visualize_coco_anns(pred_coco, idx, ax[1], shape=[W, H], alpha=0.75, draw_border=True)

        # for a in ax:
        #     a.axis('off')
        #     a.set_xlim(0, W)
        #     a.set_ylim(H, 0)

        # fig.canvas.draw()
        # fig.savefig(out_file, bbox_inches='tight', pad_inches=0)
        # plt.close(fig)



if __name__ == '__main__':

    cfg.dataset = "brightfield_coco_v2.0"
    cfg.dataset = DATASETS_CFG.get(cfg.dataset)
    
    main(cfg)

