import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.utils import nested_masks_from_list, compute_mask_iou
from utils.comm import is_dist_avail_and_initialized, get_world_size
from utils.losses import dice_loss, prob_focal_loss_jit, sigmoid_focal_loss_jit

from .matcher import HungarianMatcher, HungarianMatcherIAM

from configs import cfg


class SparseInstCriterion(nn.Module):
    # This part is partially derivated from: https://github.com/facebookresearch/detr/blob/main/models/detr.py

    def __init__(self, cfg: cfg):
        super().__init__()
        self.matcher = HungarianMatcher()

        self.losses = cfg.model.losses
        self.weight_dict = self.get_weight_dict()
        self.num_classes = cfg.model.num_classes

    def get_weight_dict(self):
        losses = ("loss_ce", "loss_bce_masks", "loss_dice_masks", "loss_objectness_masks")
        weight_dict = {}

        ce_weight = 2.0
        
        # mask.
        bce_masks_weight = 5.0
        dice_masks_weight = 2.0
        
        objectness_masks_weight = 3.0


        weight_dict = dict(
            zip(losses, (ce_weight, bce_masks_weight, dice_masks_weight, objectness_masks_weight))
            )
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

        losses = {
            # "loss_objectness": F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean'),
            # "loss_objectness_masks": nn.MSELoss()(src_iou_scores, tgt_iou_scores),
            "loss_dice_masks": dice_loss(src_masks, target_masks) / num_instances,
            "loss_bce_masks": F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean')
        }
        
        return losses 

    
    def get_loss(self, loss, outputs, targets, indices, num_instances, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks_with_iou_objectness,
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
            losses.update(self.get_loss(loss, outputs, targets, indices,
                                        num_instances, input_shape=input_shape))

        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]

        if return_matches:
            src_idx = self._get_src_permutation_idx(indices)
            tgt_idx = self._get_tgt_permutation_idx(indices)
            
            return losses, (src_idx, tgt_idx)

        return losses