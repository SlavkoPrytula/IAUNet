import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.utils import nested_masks_from_list, compute_mask_iou
from utils.comm import is_dist_avail_and_initialized, get_world_size
from utils.losses import dice_loss, prob_focal_loss_jit, sigmoid_focal_loss_jit

# from .matcher import build_matcher # HungarianMatcher, HungarianMatcherIAM

from configs import cfg

from utils.registry import CRITERIONS, MATCHERS


@CRITERIONS.register(name="SparseCriterion")
class SparseInstCriterion(nn.Module):
    # This part is partially derivated from: https://github.com/facebookresearch/detr/blob/main/models/detr.py

    def __init__(self, cfg: cfg, matcher):
        super().__init__()
        self.matcher = matcher

        self.cfg = cfg
        # self.losses = cfg.model.criterion.losses
        # self.num_classes = cfg.model.num_classes
        self.losses = cfg.losses
        self.num_classes = cfg.num_classes
        self.loss_weights = cfg.weights

        self.weight_dict = self.get_weight_dict()

    def get_weight_dict(self):
        losses = ("loss_ce", "loss_bce_masks", "loss_dice_masks", "loss_objectness_masks", 
                  "loss_overlap", "loss_duplicates", 
                  "loss_mask_iam", "loss_dice_iam",
                  "loss_ce_occluders", "loss_bce_occluders", "loss_dice_occluders", "loss_objectness_occluders", 
                  "loss_bce_overlaps", "loss_dice_overlaps",
                  "loss_masks_bounds", "loss_occluders_bounds")
        weight_dict = {}

        ce_weight = self.loss_weights.labels
        ce_occluders_weight = 2.0
        
        # mask.
        bce_masks_weight = self.loss_weights.bce_masks
        dice_masks_weight = self.loss_weights.dice_masks
        
        objectness_masks_weight = self.loss_weights.iou_masks
        objectness_occluders_weight = 1.0

        overlap_weight = 3.0
        duplicate_weight = 0.0
        iam_weight = 0.0
        iam_dice_weight = 0.0

        # occluders.
        bce_occluders_weight = 5.0
        dice_occluders_weight = 2.0

        # overlaps.
        bce_overlaps_weight = 5.0
        dice_overlaps_weight = 2.0

        # bounds.
        masks_bounds_weight = 4.0
        occluders_bounds_weight = 4.0


        weight_dict = dict(
            zip(losses, (ce_weight, bce_masks_weight, dice_masks_weight, objectness_masks_weight, 
                         overlap_weight, duplicate_weight, 
                         iam_weight, iam_dice_weight,
                         ce_occluders_weight, bce_occluders_weight, dice_occluders_weight, objectness_occluders_weight, 
                         bce_overlaps_weight, dice_overlaps_weight,
                         masks_bounds_weight, occluders_bounds_weight)))
        return weight_dict

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def loss_labels(self, outputs, targets, indices, num_instances, input_shape=None):
        assert "pred_logits" in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J]
                                     for t, (_, J) in zip(targets, indices)])
        
        # (B, N, 1)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # (B, N, 1), map labels to matched predictions
        target_classes[idx] = target_classes_o

        # (B, N, 1) -> (N, 1)
        src_logits = src_logits.flatten(0, 1)
        
        # prepare one_hot target.
        # (B, N, 1) -> (N, 1)
        # [0, 1, 1, ..., 0, 1, ...]
        target_classes = target_classes.flatten(0, 1)
        
        # TODO: check this (should be pos_inds = target_classes[target_classes == 0])
        # get positions of zero values (!= num_classes)
        pos_inds = torch.nonzero(
            target_classes != self.num_classes, as_tuple=True)[0]
       
        # create zero (N, 1) tensor and fill with 1's in pos_inds
        labels = torch.zeros_like(src_logits)
        labels[pos_inds, target_classes[pos_inds]] = 1
        

        # comp focal loss.
        class_loss = sigmoid_focal_loss_jit(
            src_logits,
            labels,
            alpha=0.25,
            gamma=2.0,
            reduction="sum",
        ) / num_instances
        losses = {'loss_ce': class_loss}
        return losses
    

    def loss_iams(self, outputs, targets, indices, num_instances, input_shape):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        # Bx100xHxW
        assert "pred_iam" in outputs
        src_masks = outputs["pred_iam"]
        
        with torch.no_grad():
            target_masks, _ = nested_masks_from_list(
                [t["iam_masks"] for t in targets], input_shape).decompose()
            
        num_masks = [len(t["iam_masks"]) for t in targets]
        
        target_masks = target_masks.to(src_masks)
        if len(target_masks) == 0:
            losses = {
                "loss_dice_iam": 0.0,
                "loss_mask_iam": 0.0
            }
            return losses
        
        src_masks = src_masks[src_idx]

        target_masks = F.interpolate(
            target_masks[:, None], size=src_masks.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

        src_masks = src_masks.flatten(1)

        # FIXME: tgt_idx
        mix_tgt_idx = torch.zeros_like(tgt_idx[1])

        cum_sum = 0
        for num_mask in num_masks:
            mix_tgt_idx[cum_sum: cum_sum + num_mask] = cum_sum
            cum_sum += num_mask
        mix_tgt_idx += tgt_idx[1]

        target_masks = target_masks[mix_tgt_idx].flatten(1)

        losses = {
            "loss_dice_iam": dice_loss(src_masks, target_masks) / num_instances,
            "loss_mask_iam": F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean')
        }
        
        return losses   


    # def loss_masks_with_iou_objectness(self, outputs, targets, indices, num_instances, input_shape):
    #     src_idx = self._get_src_permutation_idx(indices)
    #     tgt_idx = self._get_tgt_permutation_idx(indices)

    #     # Bx100xHxW
    #     assert "pred_masks" in outputs
    #     assert "pred_scores" in outputs
    #     src_iou_scores = outputs["pred_scores"]
    #     src_masks = outputs["pred_masks"]

    #     with torch.no_grad():
    #         target_masks, _ = nested_masks_from_list(
    #             [t["masks"] for t in targets], input_shape).decompose()
    #         # new
    #         prob_maps, _ = nested_masks_from_list(
    #             [t["prob_maps"] for t in targets], input_shape).decompose()
            
    #     num_masks = [len(t["masks"]) for t in targets]
        
    #     target_masks = target_masks.to(src_masks)
    #     # new
    #     prob_maps = prob_maps.to(src_masks)  

    #     if len(target_masks) == 0:
    #         losses = {
    #             "loss_dice_masks": src_masks.sum() * 0.0,
    #             "loss_bce_masks": src_masks.sum() * 0.0,
    #             # "loss_objectness": src_iou_scores.sum() * 0.0
    #         }
    #         return losses

        
    #     src_masks = src_masks[src_idx]

    #     target_masks = F.interpolate(
    #         target_masks[:, None], size=src_masks.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

        
    #     # FIXME: tgt_idx
    #     mix_tgt_idx = torch.zeros_like(tgt_idx[1])

    #     cum_sum = 0
    #     for num_mask in num_masks:
    #         mix_tgt_idx[cum_sum: cum_sum + num_mask] = cum_sum
    #         cum_sum += num_mask
    #     mix_tgt_idx += tgt_idx[1]

    #     target_masks = target_masks[mix_tgt_idx]
    #     # new
    #     prob_maps = prob_maps[mix_tgt_idx]
    #     prob_maps = prob_maps.flatten(1)

    #     # new
    #     # from utils.visualise import visualize_grid_v2

    #     # visualize_grid_v2(
    #     #     masks=src_masks.sigmoid().cpu().detach().numpy(), 
    #     #     path='./test_mask.jpg',
    #     #     ncols=5
    #     # )

    #     # visualize_grid_v2(
    #     #     masks=prob_maps.cpu().detach().numpy(), 
    #     #     path='./test_prob_map.jpg',
    #     #     ncols=5
    #     # )

    #     # raise
        
    #     src_masks = src_masks.flatten(1)
    #     target_masks = target_masks.flatten(1)
    

    #     # with torch.no_grad():
    #     #     ious = compute_mask_iou(src_masks, target_masks)
    #     # tgt_iou_scores = ious
    #     # src_iou_scores = src_iou_scores[src_idx]

    #     # tgt_iou_scores = tgt_iou_scores.flatten(0)
    #     # src_iou_scores = src_iou_scores.flatten(0)

    #     # print()
    #     # print(dice_loss(src_masks, target_masks) / num_instances)
    #     # print(F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean'))
    #     # print()
    #     # print(dice_loss(src_masks, target_masks, weights=1-prob_maps) / num_instances)
    #     # print(F.binary_cross_entropy_with_logits(src_masks, target_masks, weight=1-prob_maps, reduction='mean'))
        
    #     # raise
    #     # src_masks = src_masks.sigmoid()
    #     # src_masks = src_masks * (1 - prob_maps)

    #     losses = {
    #         # "loss_objectness": F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean'),
    #         # "loss_objectness_masks": nn.MSELoss()(src_iou_scores, tgt_iou_scores),
    #         "loss_dice_masks": dice_loss(src_masks, target_masks, weight=prob_maps) / num_instances,
    #         "loss_bce_masks": F.binary_cross_entropy_with_logits(src_masks, target_masks, weight=prob_maps, reduction='mean')
    #         # "loss_dice_masks": dice_loss(src_masks, target_masks) / num_instances,
    #         # "loss_bce_masks": F.binary_cross_entropy(src_masks, target_masks, reduction='mean')
    #     }
        
    #     return losses 


    def loss_masks_with_iou_objectness(self, outputs, targets, indices, num_instances, input_shape):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        # Bx100xHxW
        assert "pred_masks" in outputs
        assert "pred_scores" in outputs
        src_iou_scores = outputs["pred_scores"]
        src_masks = outputs["pred_masks"]

        with torch.no_grad():
            target_masks, _ = nested_masks_from_list(
                [t["masks"] for t in targets], input_shape).decompose()
            
        num_masks = [len(t["masks"]) for t in targets]
        
        target_masks = target_masks.to(src_masks)

        if len(target_masks) == 0:
            losses = {
                "loss_dice_masks": src_masks.sum() * 0.0,
                "loss_bce_masks": src_masks.sum() * 0.0,
                # "loss_objectness": src_iou_scores.sum() * 0.0
            }
            return losses

        
        src_masks = src_masks[src_idx]

        target_masks = F.interpolate(
            target_masks[:, None], size=src_masks.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

        
        # FIXME: tgt_idx
        mix_tgt_idx = torch.zeros_like(tgt_idx[1])

        cum_sum = 0
        for num_mask in num_masks:
            mix_tgt_idx[cum_sum: cum_sum + num_mask] = cum_sum
            cum_sum += num_mask
        mix_tgt_idx += tgt_idx[1]

        target_masks = target_masks[mix_tgt_idx]
        
        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)

        with torch.no_grad():
            ious = compute_mask_iou(src_masks, target_masks)
        tgt_iou_scores = ious
        src_iou_scores = src_iou_scores[src_idx]

        tgt_iou_scores = tgt_iou_scores.flatten(0)
        src_iou_scores = src_iou_scores.flatten(0)

        losses = {
            # "loss_objectness": F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean'),
            # "loss_objectness_masks": nn.MSELoss()(src_iou_scores, tgt_iou_scores),
            "loss_dice_masks": dice_loss(src_masks, target_masks) / num_instances,
            "loss_bce_masks": F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean')
        }
        
        return losses 


    # def loss_occluders(self, outputs, targets, indices, num_instances, input_shape):
    #     src_idx = self._get_src_permutation_idx(indices)
    #     tgt_idx = self._get_tgt_permutation_idx(indices)

    #     # Bx100xHxW
    #     assert "pred_occluders" in outputs
    #     src_masks = outputs["pred_occluders"]

    #     with torch.no_grad():
    #         target_masks, _ = nested_masks_from_list(
    #             [t["occluders"] for t in targets], input_shape).decompose()

    #     num_masks = [len(t["occluders"]) for t in targets]
        
    #     target_masks = target_masks.to(src_masks)
    #     if len(target_masks) == 0:
    #         losses = {
    #             "loss_overlaps": src_masks.sum() * 0.0,
    #         }
    #         return losses

    #     src_masks = src_masks[src_idx]
    #     target_masks = F.interpolate(
    #         target_masks[:, None], size=src_masks.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
    #     src_masks = src_masks.flatten(1)
        
    #     # FIXME: tgt_idx
    #     mix_tgt_idx = torch.zeros_like(tgt_idx[1])

    #     cum_sum = 0
    #     for num_mask in num_masks:
    #         mix_tgt_idx[cum_sum: cum_sum + num_mask] = cum_sum
    #         cum_sum += num_mask
    #     mix_tgt_idx += tgt_idx[1]

    #     target_masks = target_masks[mix_tgt_idx].flatten(1)

    #     losses = {
    #         # "loss_dice": dice_loss(src_masks, target_masks) / num_instances,
    #         "loss_occluders": F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean') + dice_loss(src_masks, target_masks) / num_instances,
    #     }
        
    #     return losses  


    # def _loss_labels_occluders(self, outputs, targets, indices, num_instances, input_shape, name):
    #     assert f"pred_logits_{name}" in outputs
    #     src_logits = outputs[f'pred_logits_{name}']

    #     idx = self._get_src_permutation_idx(indices)

    #     target_classes_o = torch.cat([t[f"labels_{name}"][J]
    #                                  for t, (_, J) in zip(targets, indices)])
        
    #     # (B, N, 1)
    #     target_classes = torch.full(src_logits.shape[:2], self.num_classes,
    #                                 dtype=torch.int64, device=src_logits.device)
    #     # (B, N, 1), map labels to matched predictions
    #     target_classes[idx] = target_classes_o

    #     # (B, N, 1) -> (N, 1)
    #     src_logits = src_logits.flatten(0, 1)
        
    #     # prepare one_hot target.
    #     # (B, N, 1) -> (N, 1)
    #     # [0, 1, 1, ..., 0, 1, ...]
    #     target_classes = target_classes.flatten(0, 1)
        
    #     # TODO: check this (should be pos_inds = target_classes[target_classes == 0])
    #     # get positions of zero values (!= num_classes)
    #     pos_inds = torch.nonzero(
    #         target_classes != self.num_classes, as_tuple=True)[0]
       
    #     # create zero (N, 1) tensor and fill with 1's in pos_inds
    #     labels = torch.zeros_like(src_logits)
    #     labels[pos_inds, target_classes[pos_inds]] = 1
        

    #     # comp focal loss.
    #     class_loss = sigmoid_focal_loss_jit(
    #         src_logits,
    #         labels,
    #         alpha=0.25,
    #         gamma=2.0,
    #         reduction="sum",
    #     ) / num_instances
    #     losses = {f'loss_ce_{name}': class_loss}
    #     return losses


    def _loss_mask_occluders(self, outputs, targets, indices, num_instances, input_shape, name):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        # Bx100xHxW
        assert f"pred_{name}" in outputs
        assert f"pred_scores_{name}" in outputs
        # src_iou_scores = outputs[f"pred_scores_{name}"]
        src_masks = outputs[f"pred_{name}"]

        with torch.no_grad():
            target_masks, _ = nested_masks_from_list(
                [t[name] for t in targets], input_shape).decompose()

        num_masks = [len(t[name]) for t in targets]
        
        target_masks = target_masks.to(src_masks)
        if len(target_masks) == 0:
            losses = {
                f"loss_dice_{name}": src_masks.sum() * 0.0,
                f"loss_bce_{name}": src_masks.sum() * 0.0,
                # f"loss_objectness_{name}": src_iou_scores.sum() * 0.0
            }
            return losses

        src_masks = src_masks[src_idx]
        target_masks = F.interpolate(
            target_masks[:, None], size=src_masks.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
        src_masks = src_masks.flatten(1)
        
        # FIXME: tgt_idx
        mix_tgt_idx = torch.zeros_like(tgt_idx[1])

        cum_sum = 0
        for num_mask in num_masks:
            mix_tgt_idx[cum_sum: cum_sum + num_mask] = cum_sum
            cum_sum += num_mask
        mix_tgt_idx += tgt_idx[1]

        target_masks = target_masks[mix_tgt_idx].flatten(1)

        # with torch.no_grad():
        #     ious = compute_mask_iou(src_masks, target_masks)
        # tgt_iou_scores = ious
        # src_iou_scores = src_iou_scores[src_idx]

        # tgt_iou_scores = tgt_iou_scores.flatten(0)
        # src_iou_scores = src_iou_scores.flatten(0)

        losses = {
            # "loss_dice": dice_loss(src_masks, target_masks) / num_instances,
            # f"loss_objectness_{name}": nn.MSELoss()(src_iou_scores, tgt_iou_scores),
            f"loss_dice_{name}": dice_loss(src_masks, target_masks) / num_instances,
            f"loss_bce_{name}": F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean'),
        }
        
        return losses   


    def loss_occluders(self, outputs, targets, indices, num_instances, input_shape):
        losses = {}
        # loss = self._loss_labels_occluders(outputs, targets, indices, num_instances, input_shape, name='occluders')
        # losses.update(loss)

        # for loss_name in ["masks_bounds", "occluders", "occluders_bounds"]:
        for loss_name in ["occluders"]:
        # for loss_name in ["occluders", "overlaps"]:
            loss = self._loss_mask_occluders(outputs, targets, indices, num_instances, input_shape, name=loss_name)
            losses.update(loss)

        return losses
    

    def loss_overlaps(self, outputs, targets, indices, num_instances, input_shape):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        # Bx100xHxW
        assert "pred_overlaps" in outputs
        src_masks = outputs["pred_overlaps"]

        with torch.no_grad():
            target_masks, _ = nested_masks_from_list(
                [t["overlaps"] for t in targets], input_shape).decompose()
            

        num_masks = [len(t["overlaps"]) for t in targets]
        
        target_masks = target_masks.to(src_masks)
        if len(target_masks) == 0:
            losses = {
                "loss_overlaps": src_masks.sum() * 0.0,
            }
            return losses

        
        src_masks = src_masks[src_idx]
        target_masks = F.interpolate(
            target_masks[:, None], size=src_masks.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

        src_masks = src_masks.flatten(1)

        
        # FIXME: tgt_idx
        mix_tgt_idx = torch.zeros_like(tgt_idx[1])

        cum_sum = 0
        for num_mask in num_masks:
            mix_tgt_idx[cum_sum: cum_sum + num_mask] = cum_sum
            cum_sum += num_mask
        mix_tgt_idx += tgt_idx[1]

        target_masks = target_masks[mix_tgt_idx].flatten(1)

        losses = {
            # "loss_dice": dice_loss(src_masks, target_masks) / num_instances,
            "loss_overlaps": F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean') + dice_loss(src_masks, target_masks) / num_instances,
        }
        
        return losses    
        

#     def loss_overlaps(self, outputs, targets, indices=None, num_instances=None, input_shape=None):
#         # overlaps loss
#         overlaps_src = outputs['overlaps']
        
#         overlaps_tgt = torch.cat([t["overlaps"].unsqueeze(0) for t in targets])
#         dist_maps  = torch.cat([t["dist_maps"].unsqueeze(0) for t in targets])

#         # recompute loss
#         overlaps_src = overlaps_src.sigmoid()
    
#         loss_overlap = prob_focal_loss_jit(
#                                 overlaps_src,
#                                 overlaps_tgt,
#                                 alpha=0.25,
#                                 gamma=2.0,
#                                 reduction="mean",
#                             )
# #         loss_overlap = torch.mean(loss_overlap * dist_maps)

#         losses = {
#             'loss_overlap': loss_overlap
#         }
        
#         return losses

    # def loss_overlaps(self, outputs, targets, indices, num_instances, input_shape):
    #     # overlaps loss
    #     overlaps_src = outputs['pred_overlaps']
    #     overlaps_tgt = torch.cat([t["overlaps"].unsqueeze(0) for t in targets])
        
    #     # with torch.no_grad():
    #     #     overlaps_tgt, _ = nested_masks_from_list(
    #     #         [t["overlaps"] for t in targets], input_shape).decompose()

    #     loss_overlap = sigmoid_focal_loss_jit(
    #         overlaps_src,
    #         overlaps_tgt,
    #         alpha=0.25,
    #         gamma=2.0,
    #         reduction="mean",
    #     )

    #     losses = {
    #         "loss_overlap": loss_overlap
    #     }
        
    #     return losses
    
#     def loss_duplicates(self, outputs, targets, indices=None, num_instances=None, input_shape=None):
#         pred_masks = outputs["pred_masks"]
#         pred_masks = pred_masks.sigmoid()
# #         pred_masks = pred_masks[0]
        
# #         src_idx = self._get_src_permutation_idx(indices)
# #         pred_masks = pred_masks[src_idx]
            
#         B, N, H, W = pred_masks.size()
#         duplicate_loss = 0
#         count = 0
        
#         for b in range(B):
#             masks = pred_masks[b]
            
#             # filter empty masks.
#             thr_masks = masks > 0.2
#             sum_masks = thr_masks.sum((1, 2)).float()
#             keep = sum_masks.nonzero()[:, 0]
#             masks = masks[keep]
            
#             N, _, _ = masks.size()

#             for i in range(N):
#                 for j in range(i + 1, N):
#                     loss = torch.mean(torch.abs(masks[i] - masks[j]))

#                     duplicate_loss += loss
#                     count += 1

#         duplicate_loss = duplicate_loss / count
        
#         losses = {
#             'loss_duplicates': duplicate_loss
#         }
        
#         return losses


#     def loss_duplicates(self, outputs, targets, indices=None, num_instances=None, input_shape=None):
#         # duplicate_loss = instance_kernel_loss(kernel)

#         kernel = outputs["pred_kernel"]
#         cls_logits = outputs["pred_logits"]
#         cls_logits = cls_logits.sigmoid()
        
#         B, N, D = kernel.size()

#         kernel_flat = kernel.view(B * N, D)
#         cls_logits_flat = cls_logits.view(B * N)

#         mask = cls_logits_flat > 0.1

#         obj_kernel = kernel_flat[mask]
#         bg_kernel = kernel_flat[~mask]

#         # obj_kernel = obj_kernel.view(B, obj_kernel.size(0) // B, D)
#         # bg_kernel = bg_kernel.view(B, bg_kernel.size(0) // B, D)

#         obj_sim_loss = instance_kernel_loss(obj_kernel)
#         bg_sim_loss = 1 - instance_kernel_loss(bg_kernel)
#         obj_bg_sim_loss = cosine_similarity_kernel_loss(obj_kernel, bg_kernel)

#         # edge case: no bg kernels
#         # - 0 wont work since it might push the model to remove bg completely
#         bg_sim_loss = torch.nan_to_num(bg_sim_loss, 1)
#         obj_bg_sim_loss = torch.nan_to_num(obj_bg_sim_loss, 1)
        
#         duplicate_loss = obj_sim_loss + bg_sim_loss + obj_bg_sim_loss

        
#         losses = {
#             'loss_duplicates': duplicate_loss
#         }
        
#         return losses


    def loss_duplicates(self, outputs, targets, indices, num_instances, input_shape):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)


        # Bx100xHxW
        assert "pred_masks" in outputs
        assert "pred_scores" in outputs
        src_iou_scores = outputs["pred_scores"]
        src_masks = outputs["pred_masks"]

        with torch.no_grad():
            target_masks, _ = nested_masks_from_list(
                [t["masks"] for t in targets], input_shape).decompose()
            

        num_masks = [len(t["masks"]) for t in targets]
        
        target_masks = target_masks.to(src_masks)
        if len(target_masks) == 0:
            losses = {
                "loss_dice": src_masks.sum() * 0.0,
                "loss_mask": src_masks.sum() * 0.0,
                "loss_objectness": src_iou_scores.sum() * 0.0
            }
            return losses

        
        # extracting other predictions - should correspond to bg
        src_idx = np.array(src_idx)
        bg_masks = src_masks[~src_idx]
        target_bg_masks = torch.zeros_like(bg_masks)
        target_bg_masks = target_bg_masks.to(bg_masks.device)


        losses = {
            "loss_duplicates": F.binary_cross_entropy_with_logits(bg_masks, target_bg_masks, reduction='mean') + nn.MSELoss()(bg_masks, target_bg_masks)
        }
        
        return losses
        
    
    def get_loss(self, loss, outputs, targets, indices, num_instances, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks_with_iou_objectness,
            "overlaps": self.loss_overlaps,
            "duplicates": self.loss_duplicates,
            "iam": self.loss_iams,
            "occluders": self.loss_occluders
        }
        if loss == "loss_objectness":
            # NOTE: loss_objectness will be calculated in `loss_masks_with_iou_objectness`
            return {}
        assert loss in loss_map
        return loss_map[loss](outputs, targets, indices, num_instances, **kwargs)

    
    def forward(self, outputs, targets, input_shape, return_matches=False):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, input_shape)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_instances = sum(len(t["labels"]) for t in targets)
        num_instances = torch.as_tensor([num_instances], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_instances)
        num_instances = torch.clamp(num_instances / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            # if loss == "iam":
            #     # iam guidance loss
            #     # indices_iam = self.matcher_iam(outputs_without_aux, targets, input_shape)
            #     indices_iam = indices
            #     losses.update(self.get_loss("iam", outputs, targets, indices_iam,
            #                                 num_instances, input_shape=input_shape))
            # else:
            losses.update(self.get_loss(loss, outputs, targets, indices,
                                        num_instances, input_shape=input_shape))
        
        # iam guidance loss
        # if "iam" in self.losses:
        # indices_iam = self.matcher_iam(outputs_without_aux, targets, input_shape)
        # losses.update(self.get_loss("iam", outputs, targets, indices_iam,
        #                             num_instances, input_shape=input_shape))

        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]

        if return_matches:
            src_idx = self._get_src_permutation_idx(indices)
            tgt_idx = self._get_tgt_permutation_idx(indices)
            
            return losses, (src_idx, tgt_idx)

        return losses