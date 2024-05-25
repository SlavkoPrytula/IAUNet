import gc
from os import makedirs
from os.path import join
from typing import List, Dict

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime
from itertools import islice

from utils.utils import nested_tensor_from_tensor_list
from utils.visualise import visualize, visualize_grid, visualize_grid_v2
from utils.coco.coco import COCO

from utils.evaluate.coco_evaluator import Evaluator
from visualizations.coco_vis import save_coco_vis

import wandb
from utils.logging import setup_logger
from configs import LOGGING_NAME
logger = setup_logger(name=LOGGING_NAME)


@torch.no_grad()
def valid_one_epoch(
    cfg, 
    model, 
    criterion, 
    optimizer, 
    scheduler, 
    dataloader, 
    device, 
    epoch, 
    evaluators: Dict[str, Evaluator]=None
    ):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    results = {}

    start_time = time.time()
    
    logger.info('Loss/Valid')
    # pbar = tqdm(enumerate(dataloader), total=len(dataloader), miniters=5, position=0, leave=True)
    for step, batch in enumerate(dataloader):
        if batch is None:
            continue
        
        # prepare targets
        images = []
        targets = []
        for i in range(len(batch)):
            target = batch[i]

            ignore = ["img_id", "img_path", "ori_shape", "file_name", "coco_id"]
            target = {k: v.to(cfg.device) if k not in ignore else v 
                    for k, v in target.items()}
            images.append(target["image"])

            targets.append(target)
            
        images = nested_tensor_from_tensor_list(images)   # (B, C, H, W)
        batch_size = images.tensors.size(0)
        
        pred = model(images.tensors)
        pred["img_id"] = target["img_id"]
        pred["ori_shape"] = target["ori_shape"]

        # evaluator.
        if epoch % 10 == 0:
            for evaluator_name in evaluators:
                evaluator = evaluators[evaluator_name]
                evaluator.process(pred)
        
        # get losses.
        loss_dict = criterion(pred, targets, [cfg.valid.size], epoch=epoch)
        loss = sum(loss_dict.values())
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

        if step % 10 == 0:
            mem = torch.cuda.memory_reserved() / 1E6 if torch.cuda.is_available() else 0
            current_lr = optimizer.param_groups[0]['lr']

            # eta. 
            elapsed_time = time.time() - start_time
            total_iters = len(dataloader)
            iters_done = step + 1
            iters_left = total_iters - iters_done
            avg_iter_time = elapsed_time / iters_done
            eta = avg_iter_time * iters_left
            eta = str(datetime.timedelta(seconds=int(eta)))

            logger.info(f'Epoch(valid) [{epoch}][{step}/{len(dataloader)}] loss: {epoch_loss:.4f}, eta: {eta}, lr: {current_lr:.6f}, mem: {mem:.0f}')
    
    print()
    for l in loss_dict:
        logger.info(f'{l}: {loss_dict[l]}')
    print()
    
            
    if epoch % 10 == 0:
        # evaluate.
        for evaluator_name in evaluators:
            print(f"Evaluating {evaluator_name} subset...")
            evaluator = evaluators[evaluator_name]
            evaluator.evaluate(verbose=True)
            stats = evaluator.stats

            if evaluator_name in ["valid", "eval"]:
                results.update(stats)  
                
                if wandb.run is not None:
                    for s in stats:
                        # wandb results.
                        wandb.log({f"metrics/{s}": stats[s]})

            # plot results.
            gt_coco = evaluator.gt_coco
            pred_coco = evaluator.pred_coco

            for batch in islice(dataloader, 2):
                targets = batch[0]
                
                img = targets["image"][0]
                fname = targets["file_name"]
                idx = targets["coco_id"]
                H, W = targets["ori_shape"]
                
                save_coco_vis(img, gt_coco, pred_coco, idx, shape=[H, W], 
                            path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/results/pred_[{evaluator_name}]_[{fname}].jpg')
                

    # logging results.
    results["loss_valid"] = epoch_loss
    for l in loss_dict:
        results[f"{l}_valid"] = loss_dict[l]
        
    # wandb results.
    if wandb.run is not None:
        wandb.log({f"valid/loss_valid": epoch_loss})
        for l in loss_dict:
            wandb.log({f"valid/{l}_valid": loss_dict[l]})
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return results