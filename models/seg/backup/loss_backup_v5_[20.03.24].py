import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append("./")

from utils.utils import nested_masks_from_list, nested_tensor_from_tensor_list, compute_mask_iou
from utils.comm import is_dist_avail_and_initialized, get_world_size
from utils.losses import (dice_loss, sigmoid_focal_loss_jit, 
                          sigmoid_focal_loss_hdetr, dice_loss_jit,
                          sigmoid_ce_loss_jit)

from configs import cfg
from utils.registry import CRITERIONS
from utils.visualise import visualize_grid_v2


@CRITERIONS.register(name="SparseCriterion")
class SparseInstCriterion(nn.Module):
    # This part is partially derivated from: https://github.com/facebookresearch/detr/blob/main/models/detr.py
    def __init__(self, cfg: cfg, matcher):
        super().__init__()
        self.matcher = matcher

        self.cfg = cfg
        self.losses = cfg.losses
        self.num_classes = cfg.num_classes
        self.loss_weights = cfg.weights
        self.weight_dict = self.get_weight_dict()

        self.eos_coef = self.loss_weights.no_object_weight
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)


    def get_weight_dict(self):
        losses = ("loss_ce", "loss_focal_masks", "loss_bce_masks", "loss_dice_masks", "loss_objectness_masks", 
                 "loss_ce_occluders", "loss_focal_occluders", "loss_bce_occluders", "loss_dice_occluders", "loss_objectness_occluders")
        weight_dict = {}

        ce_weight = self.loss_weights.labels
        
        # mask.
        focal_masks_weight = 0 #self.loss_weights.focal_masks
        dice_masks_weight = self.loss_weights.dice_masks
        bce_masks_weight = self.loss_weights.bce_masks
        objectness_masks_weight = self.loss_weights.iou_masks
        
        # occluders.
        ce_occluders_weight = ce_weight
        focal_occluders_weight = focal_masks_weight
        bce_occluders_weight = bce_masks_weight
        dice_occluders_weight = dice_masks_weight
        objectness_occluders_weight = objectness_masks_weight


        weight_dict = dict(
            zip(losses, (ce_weight, focal_masks_weight, bce_masks_weight, dice_masks_weight, objectness_masks_weight, 
                         ce_occluders_weight, focal_occluders_weight, bce_occluders_weight, dice_occluders_weight, objectness_occluders_weight)))
        return weight_dict
        

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    

    def loss_labels(self, outputs, targets, indices, num_masks, input_shape, **kwargs):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )  # [1, 1, 1, 1, 1, 1, ...] size(50)
        target_classes[idx] = target_classes_o # [1, 0, 1, 1, 0, 1, ...] size(50), where 0 is the obj class

        self.empty_weight = self.empty_weight.to(src_logits.device)

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
       
        # if log:
        # TODO this should probably be a separate loss, not hacked in this one here
        # losses['cls_accuracy'] = accuracy(src_logits[idx], target_classes_o, topk=100)[0]
        return losses
    

    # def loss_iou(self, outputs, targets, indices, num_masks, input_shape, **kwargs):
    #     assert "pred_masks" in outputs
    #     assert "pred_scores" in outputs

    #     src_idx = self._get_src_permutation_idx(indices)
    #     tgt_idx = self._get_tgt_permutation_idx(indices)
    #     src_masks = outputs["pred_masks"]
    #     src_masks = src_masks[src_idx]

    #     src_iou_scores = outputs["pred_scores"]

    #     masks = [t["masks"] for t in targets]
    #     target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
    #     target_masks = target_masks.to(src_masks)
    #     target_masks = target_masks[tgt_idx]

    #     src_masks = F.interpolate(src_masks[:, None], size=target_masks.shape[-2:],
    #                                 mode="bilinear", align_corners=False)

    #     src_masks = src_masks.squeeze(1)
    #     target_masks = target_masks.squeeze(1)
    #     src_masks = src_masks.flatten(1)
    #     target_masks = target_masks.flatten(1)

    #     # compute ious.
    #     with torch.no_grad():
    #         ious = compute_mask_iou(src_masks, target_masks)

    #     tgt_iou_scores = torch.full(
    #         src_iou_scores.shape[:2], 0, dtype=torch.float32, device=src_iou_scores.device
    #     )
    #     tgt_iou_scores[src_idx] = ious

    #     src_iou_scores = src_iou_scores.flatten(0)
    #     tgt_iou_scores = tgt_iou_scores.flatten(0)

    #     iou_loss = F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean')
    #     losses = {"loss_iou_masks": iou_loss}
    #     return losses


    def loss_iou(self, outputs, targets, indices, num_masks, input_shape, **kwargs):
        assert "pred_masks" in outputs
        assert "pred_scores" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]

        src_iou_scores = outputs["pred_scores"]

        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        src_masks = F.interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                    mode="bilinear", align_corners=False)

        src_masks = src_masks.squeeze(1)
        target_masks = target_masks.squeeze(1)
        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)

        # compute ious.
        with torch.no_grad():
            ious = compute_mask_iou(src_masks, target_masks)

        src_iou_scores = src_iou_scores[src_idx]
        tgt_iou_scores = ious

        src_iou_scores = src_iou_scores.flatten(0)
        tgt_iou_scores = tgt_iou_scores.flatten(0)

        iou_loss = F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean')
        losses = {"loss_iou_masks": iou_loss}
        return losses
    

    def loss_masks(self, outputs, targets, indices, num_masks, input_shape, epoch):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        assert "pred_scores" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]

        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = F.interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                    mode="bilinear", align_corners=False)

        src_masks = src_masks.squeeze(1)
        target_masks = target_masks.squeeze(1)
        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)

        losses = {
            "loss_bce_masks": sigmoid_ce_loss_jit(src_masks, target_masks, num_masks),
            "loss_dice_masks": dice_loss_jit(src_masks, target_masks, num_masks),
        }
        return losses


    def loss_occluders(self, outputs, targets, indices, num_masks, input_shape, epoch):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_occluders_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_occluders_masks"]
        src_masks = src_masks[src_idx]

        masks = [t["occluders"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = F.interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                    mode="bilinear", align_corners=False)

        src_masks = src_masks.squeeze(1)
        target_masks = target_masks.squeeze(1)
        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)

        losses = {
            "loss_bce_occluders": sigmoid_ce_loss_jit(src_masks, target_masks, num_masks),
            "loss_dice_occluders": dice_loss_jit(src_masks, target_masks, num_masks),
        }
        return losses
    

    def loss_occluders(self, outputs, targets, indices, num_masks, input_shape, epoch):
        assert "pred_occluders_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_occluders_masks"]
        src_masks = src_masks[src_idx]

        masks = [t["occluders"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = F.interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                    mode="bilinear", align_corners=False)

        src_masks = src_masks.squeeze(1)
        target_masks = target_masks.squeeze(1)
        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)

        losses = {
            "loss_bce_borders": sigmoid_ce_loss_jit(src_masks, target_masks, num_masks),
            "loss_dice_borders": dice_loss_jit(src_masks, target_masks, num_masks),
        }
        return losses
    
    
    def get_loss(self, loss, outputs, targets, indices, num_masks, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
            "occluders": self.loss_occluders,
            "borders": self.loss_borders,
            "iou": self.loss_iou,
        }
        assert loss in loss_map, f"loss {loss} not found in loss_map!"
        return loss_map[loss](outputs, targets, indices, num_masks, **kwargs)

    
    def forward(self, outputs, targets, input_shape, return_matches=False, **kwargs):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, input_shape)
        # indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices,
                                        num_masks, input_shape=input_shape, **kwargs))
        
        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]


        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # indices = self.matcher(aux_outputs, targets, input_shape)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, 
                                           num_masks, input_shape=input_shape, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)


        if return_matches:
            src_idx = self._get_src_permutation_idx(indices)
            tgt_idx = self._get_tgt_permutation_idx(indices)
            return losses, (src_idx, tgt_idx)

        return losses
    

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



if __name__ == "__main__":
    # import sys
    # sys.path.append("./")

    from models.seg.matcher import HungarianMatcher
    from utils.registry import MATCHERS
    print(MATCHERS)

    criterion = CRITERIONS.build(cfg.model.criterion)
    print(criterion)


    


