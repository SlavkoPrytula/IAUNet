from torch.cuda import amp

from utils.utils import nested_tensor_from_tensor_list
from utils.evaluate.coco_evaluator import Evaluator
from utils.callbacks import (LossLoggerCallback)
from configs import cfg

import wandb


def train_one_epoch(
        cfg: cfg, 
        model, 
        criterion,
        optimizer, 
        scheduler, 
        dataloader, 
        device, 
        epoch, 
        callbacks,
        logger,
        evaluator: Evaluator=None
        ):
    model.train()
    scaler = amp.GradScaler()
    
    results = {}
    running_loss = 0.0
    dataset_size = 0

    # TODO: this needs to be redone adn put to utild.callbacks
    total_steps = len(dataloader)
    loss_callback = LossLoggerCallback(logger, optimizer, total_steps, log_every_n_steps=10)
    callbacks.append(loss_callback)
    
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
            loss_dict, (src_idx, tgt_idx) = criterion(output, targets, [cfg.dataset.train_dataset.size], 
                                                      return_matches=True, epoch=epoch)
            loss = sum(loss_dict.values())
            
        scaler.scale(loss).backward()
        if (step + 1) % 1 == 0:
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

        for callback in callbacks:
            callback.on_train_batch_end(step=step, epoch=epoch, loss=epoch_loss)
        # loss_callback.on_train_batch_end(step, epoch, epoch_loss)

        # torch.cuda.empty_cache()
        # gc.collect()

    
    print()
    for l in loss_dict:
        logger.info(f'{l}: {loss_dict[l]}')
    print()

    # wandb results.
    # TODO: check from cfg
    if wandb.run is not None:
        wandb.log({f"train/loss_train": epoch_loss})
        for l in loss_dict:
            wandb.log({f"train/{l}_train": loss_dict[l]})

    for callback in callbacks:
        callback.on_train_epoch_end(cfg=cfg, epoch=epoch, output=output)
    # visualizer.on_train_epoch_end(cfg, epoch, output)

    # logging results.      
    results["loss_train"] = epoch_loss
    for l in loss_dict:
        results[f"{l}_train"] = loss_dict[l]
    
    return results
