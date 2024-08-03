import gc
from os import makedirs
from os.path import join
from typing import List, Dict

import torch
from itertools import islice

from utils.utils import nested_tensor_from_tensor_list
from utils.evaluate.coco_evaluator import Evaluator
from utils.callbacks import (LossLoggerCallback)
from visualizations.coco_vis import save_coco_vis
from configs import cfg

import wandb


@torch.no_grad()
def valid_one_epoch(
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
    evaluators: Dict[str, Evaluator]=None,
    ):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    results = {}

    total_steps = len(dataloader)
    loss_callback = LossLoggerCallback(logger, optimizer, total_steps, log_every_n_steps=10)
    callbacks.append(loss_callback)
    callbacks = []
    
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
            target = {k: v.to(device) if k not in ignore else v 
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
        loss_dict = criterion(pred, targets, [cfg.dataset.valid_dataset.size], epoch=epoch)
        loss = sum(loss_dict.values())
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

        for callback in callbacks:
            callback.on_valid_batch_end(step=step, epoch=epoch, loss=epoch_loss)
        # loss_callback.on_valid_batch_end(step, epoch, epoch_loss)

    print()
    for l in loss_dict:
        logger.info(f'{l}: {loss_dict[l]}')
    print()
    
    # NOTE: evaluation is now done every n epochs, this is not needed
    # check cfg.trainer.check_val_every_n_epoch
    if epoch % 10 == 0:
        makedirs(join(cfg.save_dir, 'train_visuals', f'epoch_{epoch}', 'results'), exist_ok=True)
        
        # evaluate.
        for evaluator_name in evaluators:
            print(f"Evaluating {evaluator_name} subset...")
            evaluator = evaluators[evaluator_name]
            evaluator.evaluate(verbose=True)
            stats = evaluator.stats

            if evaluator_name in ["valid", "eval"]:
                results.update(stats)  

                # TODO: check from cfg
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
    # TODO: check from cfg
    if wandb.run is not None:
        wandb.log({f"valid/loss_valid": epoch_loss})
        for l in loss_dict:
            wandb.log({f"valid/{l}_valid": loss_dict[l]})
    
    # torch.cuda.empty_cache()
    # gc.collect()
    
    return results