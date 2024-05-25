import gc

import torch
from torch import nn
from torch.cuda import amp
from tqdm import tqdm
import matplotlib.pyplot as plt


import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.seg.matcher import HungarianMatcher
from utils.utils import compute_mask_iou, flatten_mask
from utils.visualise import visualize, visualize_grid, visualize_grid_v2


from configs import cfg
from models.build_model import build_model, load_model


from dataset.datasets.brightfiled_flow import Brightfield_Dataset
from utils.normalize import normalize
from utils.augmentations import train_transforms, valid_transforms

from dataset.dataloaders import get_dataloaders
from dataset.prepare_dataset import get_folds
from dataset.datasets import df as _df

from utils.opt.matrix_nms import mask_matrix_nms

import numpy as np
from pycocotools.cocoeval import COCOeval
from utils.coco.coco import COCO
from utils.coco.mask2coco import masks2coco


np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)


def dice_score(inputs):
    N, H, W = inputs.shape
    inputs = inputs.view(N, -1)
    numerator = 2 * torch.matmul(inputs, inputs.t())
    denominator = (inputs * inputs).sum(-1)[:, None] + (inputs * inputs).sum(-1)
    score = numerator / (denominator + 1e-4)
    return score


def dice_similarity(masks):
    n = masks.shape[0]
    sim_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            intersection = np.logical_and(masks[i], masks[j]).sum()
            union = masks[i].sum() + masks[j].sum()
            sim_mat[i, j] = 2 * intersection / union
            sim_mat[j, i] = sim_mat[i, j]
    return sim_mat


def similarity_graph(A, thr=0.25):
    N = A.shape[0]
    groups = []
    visited = set()
    for i in range(N):
        if i not in visited:
            group = [i]
            visited.add(i)
            for j in range(i+1, N):
                if A[i,j] >= thr:
                    group.append(j)
                    visited.add(j)
            if len(group) > 0:
                groups.append(tuple(group))
    return groups


def remove_duplicate_nodes(G):
    visited_nodes = []
    for g in G:
        visited_nodes.extend(list(g))

    duplicate_nodes = set([x for x in visited_nodes if visited_nodes.count(x) > 1])

    filtered_nodes = []
    for g in G:
        new_g = tuple([x for x in g if x not in duplicate_nodes])
        filtered_nodes.append(new_g)

    return filtered_nodes


def mask_fusion(masks, G):
    def _fuse(masks):
        # return np.mean(masks, axis=0)

        masks = np.sum(masks, axis=0)
        masks[masks > 1] = 1
        return masks

    fused_masks = []
    for g in G:
        g = np.array(g)
        selected_masks = masks[g]
        merged_mask = _fuse(selected_masks)
        fused_masks.append(merged_mask)

    fused_masks = np.stack(fused_masks, axis=0)
    return fused_masks




# class Evaluator(nn.Module):
#     def __init__(self, cfg):
#         super(Evaluator, self).__init__()

#         self.model = load_model(cfg=cfg, path=cfg.model_weights)
#         self.model.eval()

#         self.gt_coco = {}
#         self.pred_coco = {}


#         # inference
#         self.score_threshold = 0.5 #cfg.MODEL.SPARSE_INST.CLS_THRESHOLD
#         self.mask_threshold = 0.65 #cfg.MODEL.SPARSE_INST.MASK_THRESHOLD
#         # self.max_detections = cfg.MODEL.SPARSE_INST.MAX_DETECTIONS
#         # self.num_classes = cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES

#     @torch.no_grad()
#     def inference_single(self, input):
#         output = {}
#         preds_flows, logits_cyto, preds_cyto, scores_cyto, iam = self.model(input)
        
#         output['preds_cyto'] = preds_cyto
#         output['iam'] = preds_cyto
#         output['preds_logits_cyto'] = logits_cyto
#         output['preds_scores_cyto'] = scores_cyto


#         # output = self.model(input)
#         return output


#     def evaluate(self):
#         # Create COCO evaluation object for segmentation
#         gt_coco = COCO(self.gt_coco)
#         pred_coco = COCO(self.pred_coco)
#         coco_eval = COCOeval(gt_coco, pred_coco, iouType='segm')

#         img = np.zeros((512, 512))
#         plt.figure(figsize=[10, 10])
#         annIds  = gt_coco.getAnnIds(imgIds=[1])
#         anns    = gt_coco.loadAnns(annIds)
#         plt.imshow(img)
#         gt_coco.showAnns(anns, draw_bbox=False)
#         plt.tight_layout()
#         plt.savefig('./coco_gt.jpg')


#         plt.figure(figsize=[10, 10])
#         annIds  = pred_coco.getAnnIds(imgIds=[1])
#         anns    = pred_coco.loadAnns(annIds)
#         plt.imshow(img)
#         pred_coco.showAnns(anns, draw_bbox=False)
#         plt.tight_layout()
#         plt.savefig('./coco_pred.jpg')

#         # Run the evaluation
#         coco_eval.evaluate()
#         coco_eval.accumulate()
#         coco_eval.summarize()




# class DataloaderEvaluator(Evaluator):
#     def __init__(self, cfg):
#         super(DataloaderEvaluator, self).__init__(cfg)

#         self.matcher = HungarianMatcher()


#     def forward(self, dataloader):
#         gt_masks = []
#         pred_masks = []
#         pred_scores = []
    
#         for batch in tqdm(dataloader):
            
#             bf_images, pc_images, cyto_masks, nuc_masks, cond_mask, dx_grad_masks, dy_grad_masks = batch
#             bf_images = bf_images.to(cfg.device, dtype=torch.float)
#             masks_cyto = cyto_masks.to(cfg.device)


#             output = self.inference_single(bf_images)

#             preds_cyto = output['preds_cyto']
#             preds_scores_cyto = output['preds_scores_cyto']


#             # # match
#             # indices = self.matcher(nn.Sigmoid()(preds_cyto), masks_cyto)
#             # pred_indices, gt_indices = indices[0]

#             # preds_cyto = preds_cyto[:, pred_indices, :, :]
#             # preds_scores_cyto = preds_scores_cyto[:, pred_indices, :]


#             # prepare data
#             preds_cyto = nn.Sigmoid()(preds_cyto)               # [B, N, H, W]
#             preds_scores_cyto = nn.Sigmoid()(preds_scores_cyto)   # [B, N, 1]


#             # postprocess
#             # preds_cyto = (preds_cyto > self.mask_threshold).type(torch.uint8)


#             # preds_scores_cyto = preds_scores_cyto.flatten()
#             masks_cyto = masks_cyto[0, ...]                 # (B, N, H, W) -> (N, H, W)
#             preds_cyto = preds_cyto[0, ...]                 # (B, N, H, W) -> (N, H, W)
#             preds_scores_cyto = preds_scores_cyto[0, :, 0]    # (B, N, 1) -> (N,)




#             masks = preds_cyto > self.mask_threshold

#             sum_masks = masks.sum((1, 2)).float()
#             keep = sum_masks.nonzero()[:, 0]


#             # masks = masks[keep]
#             # mask_preds = mask_preds[keep]
#             # sum_masks = sum_masks[keep]
#             # cls_scores = cls_scores[keep]
#             # cls_labels = cls_labels[keep]

#             # maskness.
#             # mask_scores = (mask_preds * masks).sum((1, 2)) / sum_masks
#             # cls_scores *= mask_scores
#             cls_labels = torch.zeros(len(masks)).to(cfg.device)        # (N,)
#             cls_scores = preds_scores_cyto


#             masks = masks[keep]
#             sum_masks = sum_masks[keep]
#             cls_scores = cls_scores[keep]
#             cls_labels = cls_labels[keep]


#             # nms
#             print()
#             print('-' * 20)
#             print('masks shape:   ', masks.shape)
#             print('scores shape: ', cls_scores.shape)
#             print('cls_labels shape: ', cls_labels.shape)
#             print(cls_labels)
#             print(cls_scores)
#             print()

#             # Matrix NMS
#             # cate_scores = mask_matrix_nms(seg_masks=seg_masks, cate_labels=cate_labels, cate_scores=cate_scores, sum_masks=sum_masks, kernel='linear')
#             scores, cls_labels, _, keep_inds = mask_matrix_nms(masks=masks, labels=cls_labels, scores=cls_scores, mask_area=sum_masks, filter_thr=-1, kernel='linear')
#             print()
#             print('keep_inds:\n', keep_inds)
#             print('scores shape: ', scores.shape)
#             print(scores)
#             print()

#             # keep = scores > self.score_threshold
#             # masks = masks[keep, :, :]
#             # scores = scores[keep]
#             # print(masks.shape)
#             # print(scores.shape)


#             # prepare data
#             masks_cyto = masks_cyto.cpu().detach().numpy()              # (N, H, W)
#             masks = masks.cpu().detach().numpy()              # (N, H, W)
#             scores = scores.cpu().detach().numpy()  # (N,)

#             masks = (masks > self.mask_threshold).astype(np.uint8)



#             sim_matrix = dice_similarity(masks)
#             print(sim_matrix)

#             sim_G = similarity_graph(sim_matrix, thr=0.25)
#             print(sim_G)

#             sim_G = remove_duplicate_nodes(sim_G)
#             print(sim_G)

#             visualize_grid_v2(
#                 masks=masks, 
#                 titles=scores, 
#                 ncols=8, 
#                 path='test_grid_plot.jpg'
#                 )
            

#             fused_masks = []
#             fused_scores = []
#             for g in sim_G:
#                 g = np.array(g)
#                 selected_masks = masks[g]
#                 selected_scores = scores[g]
                
#                 merged_mask = np.sum(selected_masks, axis=0)
#                 merged_mask[merged_mask > 1] = 1

#                 merged_scores = np.mean(selected_scores)

#                 fused_masks.append(merged_mask)
#                 fused_scores.append(merged_scores)

#             fused_masks = np.stack(fused_masks, axis=0)
#             fused_scores = np.array(fused_scores)
            
#             print(f'masks shape:        {masks.shape}')
#             print(f'fused_masks shape:  {fused_masks.shape}')

#             visualize_grid_v2(
#                 masks=fused_masks, 
#                 titles=fused_scores,
#                 ncols=8, 
#                 path='fused_grid_plot.jpg'
#                 )
            

            



#             # store data
#             gt_masks.append(masks_cyto)
#             pred_masks.append(fused_masks)

#             pred_scores.append(fused_scores)
#             break

#         # masks2coco
#         self.gt_coco = masks2coco(gt_masks)
#         self.pred_coco = masks2coco(pred_masks, scores=pred_scores)

        
#         # visualize_grid(
#         #     [20, 20], 
#         #     images=preds_cyto.cpu().detach().numpy(),
#         #     rows=10,
#         #     path=f'test_grid_plot.jpg'
#         # )



class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()

        self.model = load_model(cfg=cfg, path=cfg.model_weights)
        self.model.eval()

        self.gt_coco = {}
        self.pred_coco = {}


        # inference
        self.score_threshold = cfg.model.score_threshold #cfg.MODEL.SPARSE_INST.CLS_THRESHOLD
        self.mask_threshold = cfg.model.mask_threshold #cfg.MODEL.SPARSE_INST.MASK_THRESHOLD
        # self.max_detections = cfg.MODEL.SPARSE_INST.MAX_DETECTIONS
        # self.num_classes = cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES

    @torch.no_grad()
    def inference_single(self, input):
        output = {}
        preds_flows, logits_cyto, preds_cyto, scores_cyto, iam = self.model(input)
        
        output['preds_cyto'] = preds_cyto
        output['iam'] = preds_cyto
        output['preds_logits_cyto'] = logits_cyto
        output['preds_scores_cyto'] = scores_cyto


        # output = self.model(input)
        return output


    def evaluate(self):
        # Create COCO evaluation object for segmentation
        gt_coco = COCO(self.gt_coco)
        pred_coco = COCO(self.pred_coco)
        coco_eval = COCOeval(gt_coco, pred_coco, iouType='segm')

        img = np.zeros((512, 512))
        plt.figure(figsize=[10, 10])
        annIds  = gt_coco.getAnnIds(imgIds=[1])
        anns    = gt_coco.loadAnns(annIds)
        plt.imshow(img)
        gt_coco.showAnns(anns, draw_bbox=False)
        plt.tight_layout()
        plt.savefig('./coco_gt.jpg')


        plt.figure(figsize=[10, 10])
        annIds  = pred_coco.getAnnIds(imgIds=[1])
        anns    = pred_coco.loadAnns(annIds)
        plt.imshow(img)
        pred_coco.showAnns(anns, draw_bbox=False)
        plt.tight_layout()
        plt.savefig('./coco_pred.jpg')

        # Run the evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()




# class DataloaderEvaluator(Evaluator):
#     def __init__(self, cfg):
#         super(DataloaderEvaluator, self).__init__(cfg)

#         self.matcher = HungarianMatcher()


#     def forward(self, dataloader):
#         gt_masks = []
#         pred_masks = []
#         pred_scores = []
    
#         for batch in tqdm(dataloader):
            
#             bf_images, pc_images, cyto_masks, nuc_masks, cond_mask, dx_grad_masks, dy_grad_masks = batch
#             bf_images = bf_images.to(cfg.device, dtype=torch.float)
#             masks_cyto = cyto_masks.to(cfg.device)


#             output = self.inference_single(bf_images)

#             preds_cyto = output['preds_cyto']
#             preds_scores_cyto = output['preds_scores_cyto']


#             # prepare data
#             preds_cyto = nn.Sigmoid()(preds_cyto)               # [B, N, H, W]
#             preds_scores_cyto = nn.Sigmoid()(preds_scores_cyto)   # [B, N, 1]


#             # preds_scores_cyto = preds_scores_cyto.flatten()
#             masks_cyto = masks_cyto[0, ...]                 # (B, N, H, W) -> (N, H, W)
#             preds_cyto = preds_cyto[0, ...]                 # (B, N, H, W) -> (N, H, W)
#             preds_scores_cyto = preds_scores_cyto[0, :, 0]    # (B, N, 1) -> (N,)


#             masks = preds_cyto


#             # maskness.
#             # mask_scores = (mask_preds * masks).sum((1, 2)) / sum_masks
#             # cls_scores *= mask_scores
#             labels = torch.zeros(len(masks)).to(cfg.device)        # (N,)
#             scores = preds_scores_cyto



#             thr_masks = masks > self.mask_threshold

#             sum_masks = thr_masks.sum((1, 2)).float()
#             keep = sum_masks.nonzero()[:, 0]

#             masks = masks[keep]
#             sum_masks = sum_masks[keep]
#             scores = scores[keep]
#             labels = labels[keep]


#             # nms
#             # print()
#             # print('-' * 20)
#             # print('masks shape:   ', masks.shape)
#             # print('scores shape: ', cls_scores.shape)
#             # print('cls_labels shape: ', cls_labels.shape)
#             # print(cls_labels)
#             # print(cls_scores)
#             # print()

#             # # Matrix NMS
#             # # cate_scores = mask_matrix_nms(seg_masks=seg_masks, cate_labels=cate_labels, cate_scores=cate_scores, sum_masks=sum_masks, kernel='linear')
#             # scores, cls_labels, _, keep_inds = mask_matrix_nms(masks=masks, labels=cls_labels, scores=cls_scores, mask_area=sum_masks, filter_thr=-1, kernel='linear')
#             # print()
#             # print('keep_inds:\n', keep_inds)
#             # print('scores shape: ', scores.shape)
#             # print(scores)
#             # print()

#             # keep = scores > self.score_threshold
#             # masks = masks[keep, :, :]
#             # scores = scores[keep]
#             # print(masks.shape)
#             # print(scores.shape)



#             sim_matrix = dice_score(masks)
#             print(sim_matrix)

#             sim_G = similarity_graph(sim_matrix, thr=cfg.model.sim_score)
#             print(sim_G)

#             sim_G = remove_duplicate_nodes(sim_G)
#             print(sim_G)



#             # prepare data
#             masks_cyto = masks_cyto.cpu().detach().numpy()              # (N, H, W)
#             masks = masks.cpu().detach().numpy()              # (N, H, W)
#             scores = scores.cpu().detach().numpy()  # (N,)

#             # masks = (masks > self.mask_threshold).astype(np.uint8)


#             visualize_grid_v2(
#                 masks=masks, 
#                 titles=np.arange(0, len(masks)), 
#                 ncols=8, 
#                 path='test_grid_plot.jpg'
#                 )
            

#             fused_masks = []
#             fused_scores = []
#             for g in sim_G:
#                 g = np.array(g)
#                 selected_masks = masks[g]
#                 selected_scores = scores[g]
                
#                 # merged_mask = np.mean(selected_masks, axis=0)
#                 merged_mask = np.sum(selected_masks, axis=0)
#                 merged_mask[merged_mask > 1] = 1

#                 merged_scores = np.mean(selected_scores)

#                 fused_masks.append(merged_mask)
#                 fused_scores.append(merged_scores)

#             fused_masks = np.stack(fused_masks, axis=0)
#             fused_scores = np.array(fused_scores)
            
#             print(f'masks shape:        {masks.shape}')
#             print(f'fused_masks shape:  {fused_masks.shape}')

#             visualize_grid_v2(
#                 masks=fused_masks, 
#                 titles=fused_scores,
#                 ncols=8, 
#                 path='fused_grid_plot.jpg'
#                 )
            


#             # fused_masks = torch.tensor(fused_masks).to(cfg.device)
#             # fused_scores = torch.tensor(fused_scores).to(cfg.device)
#             # labels = torch.zeros(len(fused_masks)).to(cfg.device)

#             # # Matrix NMS
#             # thr_masks = fused_masks > self.mask_threshold
#             # sum_masks = thr_masks.sum((1, 2)).float()

#             # print(fused_scores)
#             # fused_scores, _, _, keep_inds = mask_matrix_nms(masks=fused_masks, labels=labels, scores=fused_scores, mask_area=sum_masks, filter_thr=-1, kernel='gaussian')
#             # print(fused_scores)
#             # print()

#             # keep = fused_scores > self.score_threshold
#             # fused_masks = fused_masks[keep, :, :]
#             # fused_scores = fused_scores[keep]

#             # fused_masks = fused_masks.cpu().detach().numpy()
#             # fused_scores = fused_scores.cpu().detach().numpy()


#             visualize_grid_v2(
#                 masks=fused_masks, 
#                 titles=fused_scores,
#                 ncols=8, 
#                 path='fused_nms_grid_plot.jpg'
#                 )
            
#             fused_masks = (fused_masks > self.mask_threshold).astype(np.uint8)

            

#             # store data
#             gt_masks.append(masks_cyto)
#             pred_masks.append(fused_masks)

#             pred_scores.append(fused_scores)
#             # break

#         # masks2coco
#         self.gt_coco = masks2coco(gt_masks)
#         self.pred_coco = masks2coco(pred_masks, scores=pred_scores)



class DataloaderEvaluator(Evaluator):
    def __init__(self, cfg):
        super(DataloaderEvaluator, self).__init__(cfg)

        self.matcher = HungarianMatcher()


    def forward(self, dataloader):
        gt_masks = []
        pred_masks = []
        pred_scores = []
    
        for batch in tqdm(dataloader):
            
            bf_images, pc_images, cyto_masks, nuc_masks, cond_mask, dx_grad_masks, dy_grad_masks = batch
            bf_images = bf_images.to(cfg.device, dtype=torch.float)
            masks_cyto = cyto_masks.to(cfg.device)


            output = self.inference_single(bf_images)

            preds_cyto = output['preds_cyto']
            preds_scores_cyto = output['preds_scores_cyto']
            preds_logits_cyto = output['preds_logits_cyto']


            # prepare data
            preds_cyto = nn.Sigmoid()(preds_cyto)               # [B, N, H, W]
            preds_scores_cyto = nn.Sigmoid()(preds_scores_cyto) # [B, N, 1]
            preds_logits_cyto = nn.Sigmoid()(preds_logits_cyto) # [B, N, 1]


            # preds_scores_cyto = preds_scores_cyto.flatten()
            masks_cyto = masks_cyto[0, ...]                 # (B, N, H, W) -> (N, H, W)
            preds_cyto = preds_cyto[0, ...]                 # (B, N, H, W) -> (N, H, W)
            preds_scores_cyto = preds_scores_cyto[0, :, 0]    # (B, N, 1) -> (N,)
            preds_logits_cyto = preds_logits_cyto[0, :, 0]    # (B, N, 1) -> (N,)




            # maskness.
            masks = preds_cyto
            # mask_scores = (mask_preds * masks).sum((1, 2)) / sum_masks
            # cls_scores *= mask_scores
            # labels = torch.zeros(len(masks)).to(cfg.device)        # (N,)
            labels = preds_logits_cyto
            scores = preds_scores_cyto


            # filter the empty predictions.
            thr_masks = masks > self.mask_threshold
            sum_masks = thr_masks.sum((1, 2)).float()
            keep = sum_masks.nonzero()[:, 0]

            masks = masks[keep]
            sum_masks = sum_masks[keep]
            scores = scores[keep]
            labels = labels[keep]


            # get similarity scores for every prediction and merge duplicate ones.
            sim_matrix = dice_score(masks)
            print(sim_matrix)

            sim_G = similarity_graph(sim_matrix, thr=cfg.model.sim_score)
            print(sim_G)

            sim_G = remove_duplicate_nodes(sim_G)
            print(sim_G)



            # prepare the data.
            masks_cyto = masks_cyto.cpu().detach().numpy()              # (N, H, W)
            masks = masks.cpu().detach().numpy()              # (N, H, W)
            scores = scores.cpu().detach().numpy()  # (N,)

            # masks = (masks > self.mask_threshold).astype(np.uint8)


            visualize_grid_v2(
                masks=masks, 
                titles=labels, 
                ncols=8, 
                path='test_grid_plot.jpg'
                )
            

            fused_masks = []
            fused_scores = []
            for g in sim_G:
                g = np.array(g)
                selected_masks = masks[g]
                selected_scores = scores[g]
                
                merged_mask = np.mean(selected_masks, axis=0)
                
                # merged_mask = np.sum(selected_masks, axis=0)
                # merged_mask[merged_mask > 1] = 1

                merged_scores = np.mean(selected_scores)

                # idx = np.argmax(selected_scores)
                # merged_mask = selected_masks[idx]
                # merged_scores = selected_scores[idx]

                fused_masks.append(merged_mask)
                fused_scores.append(merged_scores)

            fused_masks = np.stack(fused_masks, axis=0)
            fused_scores = np.array(fused_scores)
            
            print(f'masks shape:        {masks.shape}')
            print(f'fused_masks shape:  {fused_masks.shape}')

            visualize_grid_v2(
                masks=fused_masks, 
                titles=fused_scores,
                ncols=8, 
                path='fused_grid_plot.jpg'
                )
            


            # fused_masks = torch.tensor(fused_masks).to(cfg.device)
            # fused_scores = torch.tensor(fused_scores).to(cfg.device)
            # labels = torch.zeros(len(fused_masks)).to(cfg.device)

            # # Matrix NMS
            # thr_masks = fused_masks > self.mask_threshold
            # sum_masks = thr_masks.sum((1, 2)).float()

            # print(fused_scores)
            # fused_scores, _, _, keep_inds = mask_matrix_nms(masks=fused_masks, labels=labels, scores=fused_scores, mask_area=sum_masks, filter_thr=-1, kernel='gaussian')
            # print(fused_scores)
            # print()

            # keep = fused_scores > self.score_threshold
            # fused_masks = fused_masks[keep, :, :]
            # fused_scores = fused_scores[keep]

            # fused_masks = fused_masks.cpu().detach().numpy()
            # fused_scores = fused_scores.cpu().detach().numpy()


            visualize_grid_v2(
                masks=fused_masks, 
                titles=fused_scores,
                ncols=8, 
                path='fused_nms_grid_plot.jpg'
                )
            
            fused_masks = (fused_masks > self.mask_threshold).astype(np.uint8)

            

            # store data
            gt_masks.append(masks_cyto)
            pred_masks.append(fused_masks)

            pred_scores.append(fused_scores)
            # break

        # masks2coco
        self.gt_coco = masks2coco(gt_masks)
        self.pred_coco = masks2coco(pred_masks, scores=pred_scores)





if __name__ == "__main__":
    # 5-fold split
    df = get_folds(cfg, _df)
    print(df.groupby(['fold', 'cell_line'])['id'].count())
    train_loader, test_loader = get_dataloaders(cfg, df, fold=0)

    # test_dataset = Brightfield_Dataset(
    #     df=_df,
    #     run_type='test',
    #     img_size=cfg.valid.size,
    #     normalization=normalize,
    #     transform=valid_transforms(cfg)
    # )

    # test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=True)


    # cfg.model.arch = 'sparse_seunet_kernel_fusion'
    cfg.model.arch = 'sparse_seunet_feat_iam_mix'

    cfg.model.num_groups   = 4
    cfg.model.num_classes  = 1

    cfg.model.num_convs    = 4
    cfg.model.n_levels     = 5
    cfg.model.num_masks    = 25

    # cfg.model.num_groups   = 1
    # cfg.model.num_classes  = 1

    # cfg.model.num_convs    = 4
    # cfg.model.n_levels     = 5
    # cfg.model.num_masks    = 100


    cfg.model.score_threshold = 0.5
    cfg.model.mask_threshold = 0.5
    cfg.model.sim_score = 0.3

    # cfg.model_weights = '/gpfs/space/home/prytula/data/models/x63/experimental_segmentation/sparaseinst/sparse_unet/[seunet]_[levels_4]_[bf]_[512]_[cyto_flow]_[fixed_matching]_[removed_kernel_fusion]_[mse_obj_loss]_[weighted_loss]_[loss_1.0243].pth'
    # cfg.model_weights = '/gpfs/space/home/prytula/data/models/x63/experimental_segmentation/sparaseinst/sparse_unet/[seunet]_[levels_4]_[bf]_[512]_[cyto_flow_nuc]_[fixed_matching]_[removed_kernel_fusion]_[mse_obj_loss]_[weighted_loss]_[loss_1.4132].pth'
    # cfg.model_weights = '/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_feat_iam_mix]-[512]/added_cls_loss/checkpoints/latest.pth'

    # cfg.model_weights = '/gpfs/space/home/prytula/data/models/x63/experimental_segmentation/sparaseinst/sparse_unet/[seunet]_[levels_4]_[bf]_[512]_[cyto_flow]_[fixed_matching]_[mse_obj_loss]_[weighted_loss]_[loss_1.7457].pth'

    # mAP_50: 0.324
    # cfg.model_weights = '/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_feat_iam_mix]-[512]/[mask_loss]-[aug]/checkpoints/latest.pth'

    # mAP_50: 0.300
    # cfg.model_weights = '/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_feat_iam_mix]-[512]/[mask_loss]-[aug]2/checkpoints/latest.pth'


    # mAP_50: 0.413
    # cfg.model_weights = '/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_feat_iam_mix]-[512]/[mask_loss]-[aug]-[cls]2/checkpoints/best.pth'

    
    # mAP_50: 0.411
    cfg.model_weights = '/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_feat_iam_mix]-[512]/[mask_loss]-[aug]-[cls]3/checkpoints/best.pth'

    evaluator = DataloaderEvaluator(cfg=cfg)
    evaluator(test_loader)
    evaluator.evaluate()


