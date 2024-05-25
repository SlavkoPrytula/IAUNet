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

from configs import LOGGER

# def prepare_targets(target):
#     target = {k: v for k, v in list(target.items())[1:]}



def train_one_epoch(cfg, model, losses, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()
    
    dataset_size = 0
    running_loss = 0.0
    
    criterion = SparseInstCriterion(cfg=cfg)
    
    ncols = 5
    results = {}
    
    print('Loss/Train')
    # LOGGER.info('Loss/Train')
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # batch shape: B
        
        # prepare targets
        images = []
        targets = []
        for i in range(len(batch)):
            target = batch[i]

            target = {k: v.to(device) for k, v in target.items()}
            images.append(target["image"])

            targets.append(target)
            
        images = nested_tensor_from_tensor_list(images)   # (B, C, H, W)
        batch_size = cfg.train.bs # images.tensors.size(0)
        
        with amp.autocast(enabled=True):
            output = model(images.tensors) # (B, N, H, W)
            
            # get losses
            # TODO: return matched idxs for visualization of gt
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
        vis_preds_cyto = output['pred_masks'].sigmoid().cpu().detach().numpy()
        vis_logits_cyto  = output['pred_logits'].sigmoid().cpu().detach().numpy()
        
        visualize_grid_v2(
            masks=vis_preds_cyto[0, ...], 
            titles=vis_logits_cyto[0, :, 0],
            ncols=ncols, 
            path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/cyto.jpg'
        )
        

        # # -----------
        # # IAM Logits.  
        # iam = output['pred_iam']
        # B, N, H, W = iam.shape

        # vis_preds_iams = iam.cpu().detach().numpy()
        # # print(vis_preds_iams.min(), vis_preds_iams.max())
        # # vis_scores_cyto  = nn.Sigmoid()(output['pred_scores']).cpu().detach().numpy()
        # # vis_scores_cyto  = nn.Sigmoid()(output['pred_logits']).cpu().detach().numpy()
        
        # visualize_grid_v2(
        #     masks=vis_preds_iams[0, ...], 
        #     titles=vis_logits_cyto[0, :, 0],
        #     ncols=5, 
        #     path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/iam_logits.jpg',
        #     cmap='jet',
        #     # vmin=0, vmax=1
        # )
        

        # # -----------
        # # IAM Sigmoid.
        # vis_preds_iams = iam.clone().sigmoid().cpu().detach().numpy()
        
        # visualize_grid_v2(
        #     masks=vis_preds_iams[0, ...], 
        #     titles=vis_logits_cyto[0, :, 0],
        #     ncols=5, 
        #     path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/iam_sigmoid.jpg',
        #     cmap='jet',
        #     # vmin=0, vmax=1
        # )

        
        # # -----------
        # # IAM Softmax.  
        # iam = F.softmax(iam.view(B, N, -1), dim=-1)
        # iam = iam.view(B, N, H, W)
        # vis_preds_iams = iam.cpu().detach().numpy()
        
        # visualize_grid_v2(
        #     masks=vis_preds_iams[0, ...], 
        #     titles=vis_logits_cyto[0, :, 0],
        #     ncols=5, 
        #     path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/iam_softmax.jpg',
        #     cmap='jet',
        #     # vmin=0, vmax=1
        # )


        pred_iam = output['pred_iam']
        for k in pred_iam:
            iam = pred_iam[k]
            B, N, H, W = iam.shape
            
            # -----------
            # IAM Logits. 
            vis_preds_iams = iam.clone().cpu().detach().numpy()
            
            visualize_grid_v2(
                masks=vis_preds_iams[0, ...], 
                titles=vis_logits_cyto[0, :, 0],
                ncols=ncols, 
                path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/[{k}]_logits.jpg',
                cmap='plasma',
                # vmin=0, vmax=1
            )
            

            # -----------
            # IAM Sigmoid.
            vis_preds_iams = iam.clone().sigmoid().cpu().detach().numpy()
            
            visualize_grid_v2(
                masks=vis_preds_iams[0, ...], 
                titles=vis_logits_cyto[0, :, 0],
                ncols=ncols, 
                path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/[{k}]_sigmoid.jpg',
                cmap='plasma',
                # vmin=0, vmax=1
            )

            
            # -----------
            # IAM Softmax.  
            _iam = iam.clone()
            _iam = F.softmax(_iam.view(B, N, -1), dim=-1)
            _iam = _iam.view(B, N, H, W)
            vis_preds_iams = _iam.cpu().detach().numpy()
            
            visualize_grid_v2(
                masks=vis_preds_iams[0, ...], 
                titles=vis_logits_cyto[0, :, 0],
                ncols=ncols, 
                path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/[{k}]_softmax.jpg',
                cmap='plasma',
                # vmin=0, vmax=1
            )

        
        # -----------
        # Overlaps.
        # if "overlaps" in cfg.model.losses:
        #     vis_preds_ovlp = output['pred_overlaps'].sigmoid().cpu().detach().numpy()
        #     vis_gt_ovlp = targets[0]['overlaps'].sigmoid().cpu().detach().numpy()
            
        #     # visualize(
        #     #     [10, 5],
        #     #     preds_ovlp=vis_preds_ovlp[0, 0, ...],
        #     #     gt_ovlp=vis_gt_ovlp[0, ...],
        #     #     path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/overlap.jpg'
        #     # )
            
        #     visualize_grid_v2(
        #         masks=vis_preds_ovlp[0, ...], 
        #         titles=vis_logits_cyto[0, :, 0],
        #         ncols=ncols, 
        #         path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/overlap.jpg'
        #     )

        for loss_name in cfg.model.losses:
            if loss_name not in ["labels", "masks"]:
                vis_preds = output[f'pred_{loss_name}'].sigmoid().cpu().detach().numpy()
                vis_gt = targets[0][loss_name].sigmoid().cpu().detach().numpy()
                
                visualize_grid_v2(
                    masks=vis_preds[0, ...], 
                    titles=vis_logits_cyto[0, :, 0],
                    ncols=ncols, 
                    path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/{loss_name}.jpg'
                )
                   

        
    
    # logging results.      
    results["loss_train"] = epoch_loss
    for l in loss_dict:
        results[f"{l}_train"] = loss_dict[l]
    
    return results
