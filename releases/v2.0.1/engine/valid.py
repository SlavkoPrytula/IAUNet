import gc
from os import makedirs
from os.path import join

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from utils.utils import nested_tensor_from_tensor_list
from utils.visualise import visualize, visualize_grid, visualize_grid_v2
from utils.coco.coco import COCO

from utils.evaluate.coco_evaluator import Evaluator

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
    evaluator: Evaluator=None
    ):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    # evaluator = DataloaderEvaluator(cfg=cfg)
    results = {}
    
    print('Loss/Valid')
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), miniters=5):
        if batch is None:
            continue

        if step == 0:
            img0 = batch[0]["image"].cpu().detach().numpy()[0]
        if step == 1:
            img1 = batch[0]["image"].cpu().detach().numpy()[0]
        
        # prepare targets
        images = []
        targets = []
        for i in range(len(batch)):
            target = batch[i]

            target = {k: v.to(device) for k, v in target.items()}
            images.append(target["image"])

            targets.append(target)
            
        images = nested_tensor_from_tensor_list(images)   # (B, C, H, W)
        batch_size = images.tensors.size(0)
        
        output = model(images.tensors)
        
        # get losses
        loss_dict = criterion(output, targets, [512, 512], epoch=epoch)
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
        # makedirs(join(cfg.save_dir, 'valid_visuals', f'epoch_{epoch}'), exist_ok=True)

        # # -----------
        # # Pred Masks.
        # vis_preds_cyto = nn.Sigmoid()(output['pred_masks']).cpu().detach().numpy()
        # vis_logits_cyto  = nn.Sigmoid()(output['pred_logits']).cpu().detach().numpy()
        
        # visualize_grid_v2(
        #     masks=vis_preds_cyto[0, ...], 
        #     titles=vis_logits_cyto[0, :, 0],
        #     ncols=10, 
        #     path=f'{cfg.save_dir}/train_visuals/epoch_{epoch}/cyto.jpg'
        # )
        
        # # -----------
        # # IAM Logits.
        # vis_preds_iams = output['pred_iam'].cpu().detach().numpy()
        # # vis_scores_cyto  = nn.Sigmoid()(output['pred_scores']).cpu().detach().numpy()
        # # vis_scores_cyto  = nn.Sigmoid()(output['pred_logits']).cpu().detach().numpy()
        
        # visualize_grid_v2(
        #     masks=vis_preds_iams[0, ...], 
        #     titles=vis_logits_cyto[0, :, 0],
        #     ncols=10, 
        #     path=f'{cfg.save_dir}/valid_visuals/iam_epoch_{epoch}.jpg', 
        #     cmap='jet',
        #     # vmin=0, vmax=1
        # )
        
#         vis_preds_ovlp = nn.Sigmoid()(output['overlaps']).cpu().detach().numpy()
#         vis_gt_ovlp = nn.Sigmoid()(targets[0]['overlaps']).cpu().detach().numpy()
        
#         visualize(
#             [10, 5],
#             preds_ovlp=vis_preds_ovlp[0, 0, ...],
#             gt_ovlp=vis_gt_ovlp[0, ...],
#         )
        start_time = time.time()
        evaluator(model, dataloader)
        evaluator.evaluate(verbose=True)
        end_time = time.time()
        eval_time = end_time - start_time
        print(f"evaluation complete in {eval_time}(s)")

        results.update(evaluator.stats)

        # for i in range(1, 7):
        imgs = [img0, img1]
        for i in range(1, 3):
            # plot results.
            gt_coco = COCO(evaluator.gt_coco)
            pred_coco = COCO(evaluator.pred_coco)
            # img = np.zeros((512, 512))
            # img = images.tensors[0][i-1].cpu().detach().numpy()
            img = imgs[i-1]
            fig, ax = plt.subplots(1, 2, figsize=[20, 10])

            annIds  = gt_coco.getAnnIds(imgIds=[i])
            anns    = gt_coco.loadAnns(annIds)
            ax[0].imshow(img, cmap="gray")
            gt_coco.showAnns(anns, draw_bbox=False, ax=ax[0])
            plt.tight_layout()

            annIds  = pred_coco.getAnnIds(imgIds=[i])
            anns    = pred_coco.loadAnns(annIds)
            ax[1].imshow(img, cmap="gray")
            pred_coco.showAnns(anns, draw_bbox=False, ax=ax[1])
            plt.tight_layout()

            fig.savefig(f'{cfg.save_dir}/train_visuals/epoch_{epoch}/full_cyto_visual_{i}.jpg')
            plt.close(fig)


    # logging results.
    results["loss_valid"] = epoch_loss
    for l in loss_dict:
        results[f"{l}_valid"] = loss_dict[l]
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return results