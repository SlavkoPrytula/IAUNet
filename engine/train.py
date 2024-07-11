import gc
from os import makedirs
from os.path import join

import torch
from torch import nn
from torch.cuda import amp
from tqdm import tqdm
import numpy as np
import time
import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.utils import compute_mask_iou, flatten_mask, nested_tensor_from_tensor_list, nested_masks_from_list
from utils.visualise import visualize, visualize_grid, visualize_grid_v2
from configs import cfg

from utils.evaluate.coco_evaluator import Evaluator

import wandb
from utils.logging import setup_logger
from configs import LOGGING_NAME
logger = setup_logger(name=LOGGING_NAME)


from models.seg.loss import box_cxcywh_to_xyxy

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
    
    ncols = 5
    results = {}
    
    loss_accumulators = None
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

        if loss_accumulators is None:
            loss_accumulators = {key: 0.0 for key in loss_dict.keys()}

        for key in loss_dict:
            loss_accumulators[key] += loss_dict[key].item() * batch_size

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
    
        torch.cuda.empty_cache()
        gc.collect()

    avg_losses = {key: total / dataset_size for key, total in loss_accumulators.items()}
    
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
        makedirs(join(cfg.save_dir, 'train_visuals', f'epoch_{epoch}'), exist_ok=True)
        makedirs(join(cfg.save_dir, 'train_visuals', f'epoch_{epoch}', 'results'), exist_ok=True)


        # ===========================================
        # ========== Pred Mask Visuals ==============
        # ===========================================

        # -----------
        # Pred Masks + BBoxes.
        vis_preds_inst = output['pred_masks'].sigmoid().cpu().detach().numpy()
        probs = output['pred_logits'].softmax(-1)
        scores = probs[0, :, 0].cpu().detach().numpy()
        scores = np.round(scores, 2)

        iou_scores = output['pred_scores'].sigmoid()
        iou_scores = iou_scores[0, :, 0].cpu().detach().numpy()
        iou_scores = np.round(iou_scores, 2)

        h, w = output["pred_masks"].shape[-2:]
        bboxes = box_cxcywh_to_xyxy(output["pred_bboxes"])
        bboxes = bboxes.cpu().detach()
        bboxes = bboxes * torch.tensor([w, h, w, h], dtype=torch.float32)
        bboxes = bboxes.numpy()

        titles = [
            f"conf: {score:.2f}, iou: {iou_score:.2f}"
            for score, iou_score in zip(scores, iou_scores)
        ]
        
        visualize_grid_v2(
            masks=vis_preds_inst[0, ...], 
            bboxes=bboxes[0, ...],
            titles=titles,
            ncols=ncols, 
            path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/inst/pred_masks.jpg'
        )

        plt.figure(figsize=[10, 10])
        plt.scatter(iou_scores, scores, color="red", s=175, alpha=0.5)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("Scores", fontsize=24)
        plt.xlabel("IoU", fontsize=24)
        plt.grid(True, alpha=0.75)
        plt.tight_layout()
        plt.savefig(f'{cfg.save_dir}/train_visuals/epoch_{epoch}/iou_alignment.jpg')
        plt.close()



        # ===========================================
        # =========== Aux Mask Visuals ==============
        # ===========================================

        # -----------
        # Aux Pred Masks.
        if "aux_outputs" in output:
            for i, aux_outputs in enumerate(output['aux_outputs']):
                vis_preds_inst = aux_outputs['pred_masks'].sigmoid().cpu().detach().numpy()
                probs = aux_outputs['pred_logits'].softmax(-1)
                scores = probs[0, :, 0].cpu().detach().numpy()
                scores = np.round(scores, 2)

                iou_scores = aux_outputs['pred_scores'].sigmoid()
                iou_scores = iou_scores[0, :, 0].cpu().detach().numpy()
                iou_scores = np.round(iou_scores, 2)

                titles = [
                    f"conf: {score:.2f}, iou: {iou_score:.2f}"
                    for score, iou_score in zip(scores, iou_scores)
                ]
                
                visualize_grid_v2(
                    masks=vis_preds_inst[0, ...], 
                    titles=titles,
                    ncols=ncols, 
                    path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/aux_outputs/inst/pred_masks.jpg'
                )

                # for loss_name in ["occluders"]:
                #     vis_preds = aux_outputs[f'pred_{loss_name}_masks'].sigmoid().cpu().detach().numpy()
                #     vis_logits = aux_outputs[f'pred_{loss_name}_logits'].sigmoid().cpu().detach().numpy()
                    
                #     visualize_grid_v2(
                #         masks=vis_preds[0, ...], 
                #         titles=vis_logits[0, :, 0],
                #         ncols=ncols, 
                #         path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/aux_pred_{loss_name}_masks_{i}.jpg'
                #     )


        # -----------
        # Other.
        for inst in cfg.model.criterion.losses:
            if inst == "occluders":
                vis_preds = output['pred_occluder_masks'].sigmoid().cpu().detach().numpy()
                visualize_grid_v2(
                    masks=vis_preds[0, ...], 
                    titles=titles,
                    ncols=ncols, 
                    path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/occluders/pred_{inst}.jpg'
                )
            
            if inst == "overlaps":
                vis_preds = output['pred_overlap_masks'].sigmoid().cpu().detach().numpy()
                visualize_grid_v2(
                    masks=vis_preds[0, ...], 
                    titles=titles,
                    ncols=ncols, 
                    path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/overlaps/pred_{inst}.jpg'
                )

            if inst == "visible":
                vis_preds = output['pred_visible_masks'].sigmoid().cpu().detach().numpy()
                visualize_grid_v2(
                    masks=vis_preds[0, ...], 
                    titles=titles,
                    ncols=ncols, 
                    path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/visible/pred_{inst}.jpg'
                )

                # iam = output['pred_iam']
                # B, N, H, W = iam.shape
                
                # # -----------
                # # IAM Logits. 
                # vis_preds_iams = iam.clone().cpu().detach().numpy()
                
                # visualize_grid_v2(
                #     masks=vis_preds_iams[0, ...], 
                #     titles=titles,
                #     ncols=ncols, 
                #     path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/inst/[pred_iam]_logits.jpg',
                #     cmap='jet',
                # )

                # # -----------
                # # IAM Softmax.  
                # _iam = iam.clone()
                # _iam = F.softmax(_iam.view(B, N, -1), dim=-1)
                # _iam = _iam.view(B, N, H, W)
                # vis_preds_iams = _iam.cpu().detach().numpy()
                
                # visualize_grid_v2(
                #     masks=vis_preds_iams[0, ...], 
                #     titles=titles,
                #     ncols=ncols, 
                #     path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/inst/[pred_iam]_softmax.jpg',
                #     cmap='jet',
                # )
                
                # # -----------
                # # IAM Sigmoid.
                # vis_preds_iams = iam.clone().sigmoid().cpu().detach().numpy()
                
                # visualize_grid_v2(
                #     masks=vis_preds_iams[0, ...], 
                #     titles=titles,
                #     ncols=ncols, 
                #     path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/inst/[pred_iam]_sigmoid.jpg',
                #     cmap='jet', # plasma
                # )
            


        
        # ===========================================
        # ========== Pred IAMs Visuals ==============
        # ===========================================
        iam = output['pred_iam']
        B, N, H, W = iam.shape

        probs = output['pred_logits'].softmax(-1)
        scores = probs[0, ...].cpu().detach().numpy()
        scores = np.round(scores, 3)
        titles = [', '.join([f"({class_idx}, {score:.2f})" for class_idx, score in 
                             zip(range(scores.shape[1]), score)]) for score in scores]


        # -----------
        # IAM Logits. 
        vis_preds_iams = iam.clone().cpu().detach().numpy()
        
        visualize_grid_v2(
            masks=vis_preds_iams[0, ...], 
            titles=titles,
            ncols=ncols, 
            path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/inst/[pred_iam]_logits.jpg',
            cmap='jet',
        )

        # -----------
        # IAM Softmax.  
        _iam = iam.clone()
        _iam = F.softmax(_iam.view(B, N, -1), dim=-1)
        _iam = _iam.view(B, N, H, W)
        vis_preds_iams = _iam.cpu().detach().numpy()
        
        visualize_grid_v2(
            masks=vis_preds_iams[0, ...], 
            titles=titles,
            ncols=ncols, 
            path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/inst/[pred_iam]_softmax.jpg',
            cmap='jet',
        )
        
        # -----------
        # IAM Sigmoid.
        vis_preds_iams = iam.clone().sigmoid().cpu().detach().numpy()
        
        visualize_grid_v2(
            masks=vis_preds_iams[0, ...], 
            titles=titles,
            ncols=ncols, 
            path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/inst/[pred_iam]_sigmoid.jpg',
            cmap='jet', # plasma
        )



        n = 15
        groups = iam.shape[1] // cfg.model.instance_head.num_masks #cfg.model.instance_head.num_groups
        # -----------
        # IAM Logits [Grouped]. 
        vis_preds_iams = iam.clone().cpu().detach().numpy()

        fig, axs = plt.subplots(n, groups+1, figsize=((groups+1)*2, 30))
        for i in range(n):
            for j in range(groups+1):
                ax = axs[i, j]
                ax.axis('off')

                if j == 0:
                    ax.imshow(vis_preds_inst[0, i, :, :], cmap='viridis')
                    ax.set_title(f'pred {i}', fontsize=10)
                else:
                    channel_idx = N // groups * (j-1) + i
                    ax.imshow(vis_preds_iams[0, channel_idx, :, :], cmap='jet')
                    ax.set_title(f'head {j-1}', fontsize=10)
        
        fig.tight_layout(pad=0.5)
        plt.savefig(f'{cfg.save_dir}/train_visuals/epoch_{epoch}/inst/[pred_iam]_logits_grouped.jpg')
        plt.close()


        # -----------
        # IAM Sigmoid [Grouped]. 
        vis_preds_iams = iam.clone().sigmoid().cpu().detach().numpy()

        fig, axs = plt.subplots(n, groups+1, figsize=((groups+1)*2, 30))
        for i in range(n):
            for j in range(groups+1):
                ax = axs[i, j]
                ax.axis('off')

                if j == 0:
                    ax.imshow(vis_preds_inst[0, i, :, :], cmap='viridis')
                    ax.set_title(f'pred {i}', fontsize=10)
                else:
                    channel_idx = N // groups * (j-1) + i
                    ax.imshow(vis_preds_iams[0, channel_idx, :, :], cmap='jet')
                    ax.set_title(f'head {j-1}', fontsize=10)
        
        fig.tight_layout(pad=0.5)
        plt.savefig(f'{cfg.save_dir}/train_visuals/epoch_{epoch}/inst/[pred_iam]_sigmoid_grouped.jpg')
        plt.close()


        # -----------
        # IAM Softmax [Grouped].  
        _iam = iam.clone()
        _iam = F.softmax(_iam.view(B, N, -1), dim=-1)
        _iam = _iam.view(B, N, H, W)
        vis_preds_iams = _iam.cpu().detach().numpy()

        fig, axs = plt.subplots(n, groups+1, figsize=((groups+1)*2, 30))
        for i in range(n):
            for j in range(groups+1):
                ax = axs[i, j]
                ax.axis('off')

                if j == 0:
                    ax.imshow(vis_preds_inst[0, i, :, :], cmap='viridis')
                    ax.set_title(f'pred {i}', fontsize=10)
                else:
                    channel_idx = N // groups * (j-1) + i
                    # print(channel_idx)
                    ax.imshow(vis_preds_iams[0, channel_idx, :, :], cmap='jet')
                    ax.set_title(f'head {j-1}', fontsize=10)
        
        fig.tight_layout(pad=0.5)
        plt.savefig(f'{cfg.save_dir}/train_visuals/epoch_{epoch}/inst/[pred_iam]_softmax_grouped.jpg')
        plt.close()
        


        # ===========================================
        # =========== Aux IAMs Visuals ==============
        # ===========================================

        if "aux_outputs" in output:
            for i, aux_outputs in enumerate(output['aux_outputs']):
                iam = aux_outputs['pred_iam']
                B, N, H, W = iam.shape
                
                probs = aux_outputs['pred_logits'].softmax(-1)
                scores = probs[0, ...].cpu().detach().numpy()
                scores = np.round(scores, 3)
                titles = [', '.join([f"({class_idx}, {score:.2f})" for class_idx, score in 
                                     zip(range(scores.shape[1]), score)]) for score in scores]
                
                # -----------
                # IAM Logits. 
                vis_preds_iams = iam.clone().cpu().detach().numpy()
                
                visualize_grid_v2(
                    masks=vis_preds_iams[0, ...], 
                    titles=titles,
                    ncols=ncols, 
                    path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/aux_outputs/inst/[aux_pred_iam_{i}]_logits.jpg',
                    cmap='jet',
                )

                # -----------
                # IAM Softmax.  
                _iam = iam.clone()
                _iam = F.softmax(_iam.view(B, N, -1), dim=-1)
                _iam = _iam.view(B, N, H, W)
                vis_preds_iams = _iam.cpu().detach().numpy()
                
                visualize_grid_v2(
                    masks=vis_preds_iams[0, ...], 
                    titles=titles,
                    ncols=ncols, 
                    path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/aux_outputs/inst/[aux_pred_iam_{i}]_softmax.jpg',
                    cmap='jet',
                )
                
                # -----------
                # IAM Sigmoid.
                vis_preds_iams = iam.clone().sigmoid().cpu().detach().numpy()
                
                visualize_grid_v2(
                    masks=vis_preds_iams[0, ...], 
                    titles=titles,
                    ncols=ncols, 
                    path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/aux_outputs/inst/[aux_pred_iam_{i}]_sigmoid.jpg',
                    cmap='jet', # plasma
                )
    

    # logging results.      
    results["loss_train"] = epoch_loss
    for l in loss_dict:
        results[f"{l}_train"] = loss_dict[l]
    
    return results
