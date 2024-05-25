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
    
    ncols = 5
    
    print('Loss/Train')
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
        # save results here

    return epoch_loss
