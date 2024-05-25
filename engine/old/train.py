import gc

import torch
from torch import nn
from torch.cuda import amp
from tqdm import tqdm

import torch.nn.functional as F

from models.seg.matcher import HungarianMatcher
from utils.losses import DiceLoss, FocalLoss
from utils.utils import compute_mask_iou, flatten_mask
from utils.visualise import visualize, visualize_grid, visualize_grid_v2

from models.seg.loss import SparseInstCriterion


def train_one_epoch(cfg, model, losses, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()
    
    dataset_size = 0
    running_loss = 0.0

    criterion = SparseInstCriterion()

    metrics = ['Loss/Train', 'Loss/Train BCE (cyto) loss', 'Loss/Train Dice (cyto) loss', 'Loss/Train BCE2 (cyto) loss', 'Loss/Train Obj (cyto) loss', 'Loss/Train Cls (cyto) loss']

    
    for step, batch in enumerate(dataloader):
        bf_images, pc_images, cyto_masks, nuc_masks, cond_mask = batch
        bf_images = bf_images.to(device, dtype=torch.float)
        # pc_images = pc_images.to(device, dtype=torch.float)
        masks_cyto = cyto_masks.to(device)
        # masks_nuc = nuc_masks.to(device)
        cond_mask = cond_mask.to(device)


        batch_size = bf_images.size(0)
        
        with amp.autocast(enabled=True):
            output = model(bf_images)
            print(output.shape)
            
            targets = []
            for i in range(batch_size):
                N, _, _ = masks_cyto[i].shape

                gt_labels = torch.ones(N, dtype=torch.long).to(device)
                
                target = {
                    'labels': gt_labels,
                    'masks': masks_cyto[0]
                }
                targets.append(target)

            losses = criterion(output, targets, [512, 512])

            print(losses)
            


            
        scaler.scale(loss).backward()
        if (step + 1) % 1 == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        torch.cuda.empty_cache()
        gc.collect()


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
            path=f'{cfg.save_dir}/train_visuals/cyto_epoch_{epoch}.jpg'
        )


        # visualize_grid(
        #     [20, 20], 
        #     images=vis_preds_cyto[0, ...],
        #     rows=10,
        #     path=f'{cfg.save_dir}/train_visuals/cyto_epoch_{epoch}.jpg'
        # )

        # visualize_grid(
        #     [20, 20], 
        #     images=vis_preds_nuc[0, ...],
        #     rows=10,
        #     path=f'{cfg.save_dir}/train_visuals/iam_epoch_{epoch}.jpg'
        # )
        
        
        # preds_flows = nn.Sigmoid()(preds_flows).cpu().detach().numpy()
        
        # # visualize(
        # #     [10, 5], 
        # #     dx_grad_preds=preds_flows[0, 0, ...], 
        # #     dy_grad_preds=preds_flows[0, 1, ...],
        # #     path=f'{cfg.save_dir}/train_visuals/flows_epoch_{epoch}.jpg'
        # #     )
        # visualize(
        #     [10, 5],
        #     pred_mask=preds_flows[0, 0, ...], 
        #     # dy_grad_preds=preds_flows[0, 1, ...],
        #     path=f'{cfg.save_dir}/train_visuals/mask_epoch_{epoch}.jpg'
        #     )
        

    return epoch_loss

