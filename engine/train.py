import torch
from torch.cuda import amp
import time
import datetime

from utils.utils import nested_tensor_from_tensor_list
from configs import cfg
from utils.evaluate.coco_evaluator import Evaluator

import wandb
from utils.logging import setup_logger
from configs import LOGGING_NAME
logger = setup_logger(name=LOGGING_NAME)

from utils.registry import VISUALIZERS
visualizer = VISUALIZERS.build(cfg.visualizer)


def train_one_epoch(
        cfg: cfg, 
        model, 
        criterion,
        optimizer, 
        scheduler, 
        dataloader, 
        device, 
        epoch, 
        evaluator: Evaluator=None
        ):
    model.train()
    scaler = amp.GradScaler()
    
    results = {}
    
    running_loss = 0.0
    dataset_size = 0

    start_time = time.time()
    
    logger.info('Loss/Train')
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
            target = {k: v.to(device) if k not in ignore else v 
                    for k, v in target.items()}
            images.append(target["image"])
            targets.append(target)
            
        images = nested_tensor_from_tensor_list(images)   # (B, C, H, W)
        batch_size = images.tensors.size(0)
        
        with amp.autocast(enabled=True):
            output = model(images.tensors) # (B, N, H, W)
            
            # get losses
            loss_dict, (src_idx, tgt_idx) = criterion(output, targets, [cfg.train.size], 
                                                      return_matches=True, epoch=epoch)
            loss = sum(loss_dict.values())
            
        scaler.scale(loss).backward()
        if (step + 1) % 1 == 0:
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()
                
            optimizer.zero_grad()
                
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

            logger.info(f'Epoch(train) [{epoch}][{step}/{len(dataloader)}] loss: {epoch_loss:.4f}, eta: {eta}, lr: {current_lr:.6f}, mem: {mem:.0f}')
    
        # torch.cuda.empty_cache()
        # gc.collect()
    
    print()
    for l in loss_dict:
        logger.info(f'{l}: {loss_dict[l]}')
    print()

    # wandb results.
    if wandb.run is not None:
        wandb.log({f"train/loss_train": epoch_loss})
        for l in loss_dict:
            wandb.log({f"train/{l}_train": loss_dict[l]})

    
    if epoch % 10 == 0:
        visualizer.on_train_epoch_end(cfg, epoch, output)

    # logging results.      
    results["loss_train"] = epoch_loss
    for l in loss_dict:
        results[f"{l}_train"] = loss_dict[l]
    
    return results
