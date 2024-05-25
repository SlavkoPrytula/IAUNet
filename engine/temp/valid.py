import gc
from os import makedirs
from os.path import join

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from utils.utils import nested_tensor_from_tensor_list
from utils.visualise import visualize, visualize_grid, visualize_grid_v2

from utils.evaluate.dataloader_evaluator import DataloaderEvaluator
from models.seg.loss import SparseInstCriterion


@torch.no_grad()
def valid_one_epoch(cfg, model, losses, optimizer, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    criterion = SparseInstCriterion(cfg=cfg)
    evaluator = DataloaderEvaluator(cfg=cfg)
    
    print('Loss/Valid')
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # prepare targets
        images = []
        targets = []
        for i in range(len(batch)):
            target = batch[i]

            target = {k: v.to(device) for k, v in target.items()}
            images.append(target["image"])

            targets.append(target)
            
        images = nested_tensor_from_tensor_list(images)   # (B, C, H, W)
        batch_size = cfg.valid.bs # images.tensors.size(0)
        
        output = model(images.tensors)
        
        # get losses
        loss_dict = criterion(output, targets, [512, 512])
        loss = sum(loss_dict.values())
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size
        
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']

    print()
    print(f'epoch_loss: {epoch_loss:.4f}')
    print()
        
    
    for l in loss_dict:
        print(f'{l}: {loss_dict[l]}')
            
    if epoch % 10 == 0:
        # save results here
        
        evaluator(model, dataloader)
        evaluator.evaluate()
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return epoch_loss