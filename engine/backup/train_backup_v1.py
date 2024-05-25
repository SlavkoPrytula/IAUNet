import gc
from os import makedirs
from os.path import join

import torch
from torch import nn
from torch.cuda import amp
from tqdm import tqdm

import torch.nn.functional as F

from utils.utils import compute_mask_iou, flatten_mask, nested_tensor_from_tensor_list
from utils.visualise import visualize, visualize_grid, visualize_grid_v2

from models.seg.loss import SparseInstCriterion


def train_one_epoch(cfg, model, losses, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()
    
    dataset_size = 0
    running_loss = 0.0
    
    criterion = SparseInstCriterion(cfg=cfg)
    
    print('Loss/Train')
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # batch shape: B
        
        # prepare targets
        images = []
        targets = []
        for i in range(len(batch)):
            bf_images, cyto_masks, masks_iam = batch[i]
            bf_images = bf_images.to(device, dtype=torch.float)
            masks_cyto = cyto_masks.to(device)
            masks_iam = masks_iam.to(device)
            
            # image
            images.append(bf_images)
            
            # labels
            N, _, _ = masks_cyto.shape
            gt_labels = torch.zeros(N, dtype=torch.int64)
            gt_labels = gt_labels.to(device)

            N, _, _ = masks_iam.shape
            gt_labels_iam = torch.zeros(N, dtype=torch.int64)
            gt_labels_iam = gt_labels_iam.to(device)

            # targets
            target = {
                'labels': gt_labels,
                'masks': masks_cyto,
                'iam_labels': gt_labels_iam,
                'iam_masks': masks_iam
            }
            targets.append(target)
            
        images = nested_tensor_from_tensor_list(images)   # (B, C, H, W)


        batch_size = cfg.train.bs # images.tensors.size(0)
        
        with amp.autocast(enabled=True):
            output = model(images.tensors) # (B, N, H, W)
            
            # get losses
            loss_dict = criterion(output, targets, [512, 512])
            loss = sum(loss_dict.values())
            
            
        scaler.scale(loss).backward()
        if (step + 1) % 1 == 0:
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()
                
            # zero the parameter gradients
            optimizer.zero_grad()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        torch.cuda.empty_cache()
        gc.collect()
        
    print()
    print(f'epoch_loss: {epoch_loss:.4f}')
    print()
    
    
    for l in loss_dict:
        print(f'{l}: {loss_dict[l]}')
    
    if epoch % 10 == 0:
        makedirs(join(cfg.save_dir, 'train_visuals', f'epoch_{epoch}'), exist_ok=True)


        # -----------
        # Pred Masks.
        vis_preds_cyto = nn.Sigmoid()(output['pred_masks']).cpu().detach().numpy()
        vis_logits_cyto  = nn.Sigmoid()(output['pred_logits']).cpu().detach().numpy()
        
        visualize_grid_v2(
            masks=vis_preds_cyto[0, ...], 
            titles=vis_logits_cyto[0, :, 0],
            ncols=10, 
            path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/cyto.jpg'
        )
        

        # -----------
        # IAM Logits.  
        iam = output['pred_iam']
        B, N, H, W = iam.shape

        vis_preds_iams = iam.cpu().detach().numpy()
        # print(vis_preds_iams.min(), vis_preds_iams.max())
        # vis_scores_cyto  = nn.Sigmoid()(output['pred_scores']).cpu().detach().numpy()
        # vis_scores_cyto  = nn.Sigmoid()(output['pred_logits']).cpu().detach().numpy()
        
        visualize_grid_v2(
            masks=vis_preds_iams[0, ...], 
            titles=vis_logits_cyto[0, :, 0],
            ncols=10, 
            path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/iam_logits.jpg',
            cmap='jet',
            # vmin=0, vmax=1
        )
        

        # -----------
        # IAM Sigmoid.
        vis_preds_iams = iam.clone().sigmoid().cpu().detach().numpy()
        
        visualize_grid_v2(
            masks=vis_preds_iams[0, ...], 
            titles=vis_logits_cyto[0, :, 0],
            ncols=10, 
            path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/iam_sigmoid.jpg',
            cmap='jet',
            # vmin=0, vmax=1
        )

        
        # -----------
        # IAM Softmax.  
        iam = F.softmax(iam.view(B, N, -1), dim=-1)
        iam = iam.view(B, N, H, W)
        vis_preds_iams = iam.cpu().detach().numpy()
        
        visualize_grid_v2(
            masks=vis_preds_iams[0, ...], 
            titles=vis_logits_cyto[0, :, 0],
            ncols=10, 
            path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/iam_softmax.jpg',
            cmap='jet',
            # vmin=0, vmax=1
        )


        
#         vis_preds_ovlp = nn.Sigmoid()(output['overlaps']).cpu().detach().numpy()
#         vis_gt_ovlp = nn.Sigmoid()(targets[0]['overlaps']).cpu().detach().numpy()
        
#         visualize(
#             [10, 5],
#             preds_ovlp=vis_preds_ovlp[0, 0, ...],
#             gt_ovlp=vis_gt_ovlp[0, ...],
#         )
        
        
    return epoch_loss
