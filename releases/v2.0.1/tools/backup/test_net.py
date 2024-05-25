import gc

import torch
from torch import nn
from torch.cuda import amp
from tqdm import tqdm


import torch.nn.functional as F

from models.seg.matcher import HungarianMatcher
from utils.utils import compute_mask_iou, flatten_mask
from utils.visualise import visualize, visualize_grid


from configs import cfg
from models.build_model import build_model, load_model

from dataset.dataloaders import get_dataloaders
from dataset.prepare_dataset import get_folds
from dataset.datasets import df as _df


import numpy as np
from pycocotools.cocoeval import COCOeval

from utils.coco.coco import COCO
from utils.coco.mask2coco import masks2coco




model_weights = '/gpfs/space/home/prytula/data/models/x63/experimental_segmentation/sparaseinst/sparse_unet/[seunet]_[levels_4]_[bf]_[512]_[cyto_flow]_[fixed_matching]_[removed_kernel_fusion]_[mse_obj_loss]_[weighted_loss]_[loss_1.0243].pth'
model = load_model(cfg=cfg, path=model_weights)
# print(model)


# 5-fold split
df = get_folds(cfg, _df)
print(df.groupby(['fold', 'cell_line'])['id'].count())
train_loader, valid_loader = get_dataloaders(cfg, df, fold=0)

gt_masks = []
pred_masks = []
pred_scores = []


matcher = HungarianMatcher()

model.eval()
with torch.no_grad():
    for _ in tqdm(range(5)):
        batch = next(iter(valid_loader))
        
        bf_images, pc_images, cyto_masks, nuc_masks, cond_mask, dx_grad_masks, dy_grad_masks = batch
        bf_images = bf_images.to(cfg.device, dtype=torch.float)
        pc_images = pc_images.to(cfg.device, dtype=torch.float)
        masks_cyto = cyto_masks.to(cfg.device)
        masks_nuc = nuc_masks.to(cfg.device)
        cond_mask = cond_mask.to(cfg.device)

        dx_grad_masks = dx_grad_masks.to(cfg.device)
        dy_grad_masks = dy_grad_masks.to(cfg.device)
        masks_flows = torch.cat([dx_grad_masks, dy_grad_masks], 1)

        preds_flows, logits_cyto, preds_cyto, scores_cyto, iam = model(bf_images)



        # match
        indices = matcher(nn.Sigmoid()(preds_cyto), masks_cyto)
        pred_indices, gt_indices = indices[0]

        preds_cyto = preds_cyto[:, pred_indices, :, :]


        # prepare data
        iam = nn.Sigmoid()(iam)
        iam = iam.cpu().detach().numpy()  # [B, N, H, W]

        masks_cyto = masks_cyto.cpu().detach().numpy()  # [B, N, H, W]

        preds_cyto = nn.Sigmoid()(preds_cyto)
        preds_cyto = preds_cyto.cpu().detach().numpy()  # [B, N, H, W]

        scores_cyto = nn.Sigmoid()(scores_cyto)
        scores_cyto = scores_cyto.cpu().detach().numpy()  # [B, N, H, W]

        logits_cyto = nn.Sigmoid()(logits_cyto)
        logits_cyto = logits_cyto.cpu().detach().numpy()  # [B, N, H, W]

        preds_flows = nn.Sigmoid()(preds_flows)
        preds_flows = preds_flows.cpu().detach().numpy()  # [B, N, H, W]


        # visualize_grid(
        #     [20, 20], 
        #     images=preds_cyto[0, ...],
        #     rows=10,
        #     path='test.jpg'
        # )


        # postprocess
        preds_cyto = (preds_cyto > 0.5).astype(np.uint8)

        masks_cyto = masks_cyto[0, ...]
        preds_cyto = preds_cyto[0, ...]

        # store data
        gt_masks.append(masks_cyto)
        pred_masks.append(preds_cyto)

        # dummmy prediction scores
        _pred_scores = np.ones(len(preds_cyto))
        pred_scores.append(_pred_scores)


for i in gt_masks:
    print(i.shape)

for i in pred_masks:
    print(i.shape)

for i in pred_scores:
    print(i.shape)

# masks2coco
gt_coco = masks2coco(gt_masks)
pred_coco = masks2coco(pred_masks, scores=pred_scores)


# Create COCO evaluation object for segmentation
gt_coco = COCO(gt_coco)
pred_coco = COCO(pred_coco)
coco_eval = COCOeval(gt_coco, pred_coco, iouType='segm')

# Run the evaluation
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()


