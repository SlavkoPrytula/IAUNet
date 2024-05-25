import gc

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import torch.nn.functional as F

from models.seg.matcher import HungarianMatcher
from utils.losses import DiceLoss, FocalLoss
from utils.utils import compute_mask_iou, flatten_mask
from utils.visualise import visualize, visualize_grid, visualize_grid_v2



@torch.no_grad()
def valid_one_epoch(cfg, model, losses, optimizer, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    val_scores = []
    
    matcher = HungarianMatcher()
    
    # metrics = ['Loss/Valid', 'Loss/Valid BCE (cyto) loss', 'Loss/Valid Dice (cyto) loss', 'Loss/Valid MSE (flow) loss', 'Loss/Valid Obj (cyto) loss',
    #            'Loss/Valid BCE (nuc) loss', 'Loss/Valid Dice (nuc) loss']

    metrics = ['Loss/Valid', 'Loss/Valid BCE (cyto) loss', 'Loss/Valid Dice (cyto) loss', 'Loss/Valid BCE2 (cyto) loss', 'Loss/Valid Obj (cyto) loss', 'Loss/Valid Cls (cyto) loss']

    
    for step, batch in enumerate(dataloader):                
        bf_images, pc_images, cyto_masks, nuc_masks, cond_mask, dx_grad_masks, dy_grad_masks = batch
        bf_images = bf_images.to(device, dtype=torch.float)
        pc_images = pc_images.to(device, dtype=torch.float)
        masks_cyto = cyto_masks.to(device)
        masks_nuc = nuc_masks.to(device)
        cond_mask = cond_mask.to(device)
        
        dx_grad_masks = dx_grad_masks.to(device)
        dy_grad_masks = dy_grad_masks.to(device)
        masks_flows = torch.cat([dx_grad_masks, dy_grad_masks], 1)
        
        _, N, _, _ = masks_cyto.shape
        batch_size = bf_images.size(0)
        
        preds_flows, logits_cyto, preds_cyto, scores_cyto, preds_nuc = model(bf_images)
            
        # ==== Cyto ====
        indices = matcher(nn.Sigmoid()(preds_cyto), masks_cyto)
        
        vis_preds_cyto = nn.Sigmoid()(preds_cyto).cpu().detach().numpy().copy()
        vis_logits_cyto = nn.Sigmoid()(logits_cyto).cpu().detach().numpy().copy()
        vis_preds_nuc = nn.Sigmoid()(preds_nuc).cpu().detach().numpy().copy()



        # indexing matched predictions
        pred_indices, gt_indices = indices[0]
        preds_cyto = preds_cyto[0, pred_indices, :, :]
        masks_cyto = masks_cyto[0, gt_indices, :, :]
        
        
        # Losses
        # bce loss
        bce_loss = F.binary_cross_entropy_with_logits(preds_cyto, masks_cyto, reduction='mean')
        # dice loss
        dice_loss = DiceLoss(nn.Sigmoid()(preds_cyto), masks_cyto) / N
        # reconstruction loss
        # mse_loss = nn.MSELoss()(preds_flows, masks_flows)
        mask_loss = F.binary_cross_entropy_with_logits(preds_flows, cond_mask, reduction='mean')


        # iou loss
        pred_iou = scores_cyto[0, pred_indices, :]
        with torch.no_grad():
            tg_ious = compute_mask_iou(nn.Sigmoid()(preds_cyto).flatten(1), masks_cyto.flatten(1))

        # pred_iou = scores_cyto[0, pred_indices, :]
        # pred_iou = scores_cyto[0]
        # with torch.no_grad():
        #     tg_ious = torch.zeros(100).to(device)
        #     _tg_ious = compute_mask_iou(nn.Sigmoid()(preds_cyto).flatten(1), masks_cyto.flatten(1))
        #     _tg_ious = _tg_ious.to(device)
        #     tg_ious[pred_indices] = _tg_ious

        pred_iou = pred_iou.float().flatten(0)
        tg_ious = tg_ious.float().flatten(0)

        obj_loss = F.binary_cross_entropy_with_logits(pred_iou, tg_ious, reduction='mean')
        # obj_loss = nn.MSELoss()(nn.Sigmoid()(pred_iou), tg_ious)
        

        # class loss.
        labels = torch.zeros(25, 1)
        labels[pred_indices] = 1
        labels = labels.to(device)

        # logits_cyto = logits_cyto[0, pred_indices, :]
        cls_loss = FocalLoss(logits_cyto, labels, alpha=0.25, gamma=2.0, reduction="mean")
        
        
        
        # # ==== Nuc ====
        # # target instance actiavation maps
        # # guide them to output nucleus

        # # preds_cyto shape: (B, N, H, W)
        # # preds_nuc shape:  (B, N, H, W)

        # _, N, _, _ = masks_nuc.shape

        # indices = matcher(nn.Sigmoid()(preds_nuc), masks_nuc)
        # pred_indices, gt_indices = indices[0]

        # preds_nuc = preds_nuc[0, pred_indices, :, :]
        # masks_nuc = masks_nuc[0, gt_indices, :, :]

        # bce_loss_nuc = F.binary_cross_entropy_with_logits(preds_nuc, masks_nuc, reduction='mean')
        # dice_loss_nuc = DiceLoss(nn.Sigmoid()(preds_nuc), masks_nuc) / N


        # bce_loss_nuc = 3 * bce_loss_nuc
        # dice_loss_nuc = 3 * dice_loss_nuc
        
        bce_loss = 5 * bce_loss
        dice_loss = 2 * dice_loss
        mask_loss = mask_loss
        obj_loss = 3 * obj_loss
        cls_loss = 2 * cls_loss

        loss = bce_loss + dice_loss + mask_loss + obj_loss + cls_loss #+ bce_loss_nuc + dice_loss_nuc
        
        
        
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']

    if epoch % 1 == 0:
        # get current losses
        # vals = [epoch_loss, bce_loss, dice_loss, mse_loss, obj_loss, bce_loss_nuc, dice_loss_nuc]
        # s = ('%28s,' * 8 % tuple(['epoch'] + metrics)).rstrip(',') + '\n'
        # print(s + ('%28.5g,' * 8 % tuple([epoch] + vals)).rstrip(',') + '\n')

        vals = [epoch_loss, bce_loss, dice_loss, mask_loss, obj_loss, cls_loss]
        s = ('%28s,' * 7 % tuple(['epoch'] + metrics)).rstrip(',') + '\n'
        print(s + ('%28.5g,' * 7 % tuple([epoch] + vals)).rstrip(',') + '\n')
    
        
    if epoch % 10 == 0:
        visualize_grid_v2(
            masks=vis_preds_cyto[0, ...], 
            titles=vis_logits_cyto[0, :, 0],
            ncols=8, 
            path=f'{cfg.save_dir}/valid_visuals/cyto_epoch_{epoch}.jpg'
        )

        # visualize_grid(
        #     [20, 20], 
        #     images=vis_preds_cyto[0, ...],
        #     rows=10,
        #     path=f'{cfg.save_dir}/valid_visuals/cyto_epoch_{epoch}.jpg'
        # )

        # visualize_grid(
        #     [20, 20], 
        #     images=vis_preds_nuc[0, ...],
        #     rows=10,
        #     path=f'{cfg.save_dir}/valid_visuals/iam_epoch_{epoch}.jpg'
        # )

        
        # preds_flows = nn.Sigmoid()(preds_flows).cpu().detach().numpy()
        
        # visualize(
        #     [10, 5],
        #     pred_mask=preds_flows[0, 0, ...], 
        #     # dy_grad_preds=preds_flows[0, 1, ...],
        #     path=f'{cfg.save_dir}/valid_visuals/mask_epoch_{epoch}.jpg'
        #     )
    
    val_scores = np.mean(val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss