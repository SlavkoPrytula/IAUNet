import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append("./")

from utils.utils import nested_tensor_from_tensor_list, compute_mask_iou
from utils.dist.comm import is_dist_avail_and_initialized, get_world_size
from utils.losses import sigmoid_ce_loss_jit, dice_loss_jit, sigmoid_focal_loss
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from configs import cfg
from utils.registry import CRITERIONS
from .matcher import point_sample


def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))



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

        self.num_points = 112 * 112
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75

        self.focal_alpha = 0.25
        self.semantic_ce_loss = True

        # print(self.matcher)
        # print(self.weight_dict)
        # print(f'\ncomputing loss for {self.num_classes} classes\n')
        print(self)


    def get_weight_dict(self):
        losses = ("loss_ce", "loss_bce_masks", "loss_dice_masks", "loss_objectness_masks", 
                  "loss_giou", "loss_bbox",)
        weight_dict = {}

        # cls.
        ce_weight = self.loss_weights.labels
        
        # mask.
        # focal_masks_weight = 0 #self.loss_weights.focal_masks
        dice_masks_weight = self.loss_weights.dice_masks
        bce_masks_weight = self.loss_weights.bce_masks
        objectness_masks_weight = self.loss_weights.iou_masks
        giou_weight = self.loss_weights.giou_weight
        bbox_weight = self.loss_weights.bbox_weight
        
        weight_dict = dict(
            zip(losses, (
                ce_weight, bce_masks_weight, dice_masks_weight, objectness_masks_weight, 
                giou_weight, bbox_weight,))
        )
        
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
    

    def loss_labels_ce(self, outputs, targets, indices, num_masks, **kwargs):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        self.empty_weight = self.empty_weight.to(src_logits.device)

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses


    def loss_labels(self, outputs, targets, indices, num_masks, **kwargs):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_masks, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        return losses


    def loss_iou(self, outputs, targets, indices, num_masks, **kwargs):
        assert "pred_masks" in outputs, f"masks not found in losses."
        assert "pred_scores" in outputs, f"scores not found."

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
    

    # def _loss_masks(self, outputs, targets, indices, num_masks, name, loss_type="bce"):
    #     assert f"pred_{name}" in outputs, f"{name} not found in losses."

    #     src_idx = self._get_src_permutation_idx(indices)
    #     tgt_idx = self._get_tgt_permutation_idx(indices)
    #     src_masks = outputs[f"pred_{name}"]
    #     src_masks = src_masks[src_idx]

    #     masks = [t[f"{name}"] for t in targets]
    #     # TODO use valid to mask invalid areas due to padding in loss
    #     target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
    #     target_masks = target_masks.to(src_masks)
    #     target_masks = target_masks[tgt_idx]

    #     # No need to upsample predictions as we are using normalized coordinates :)
    #     # N x 1 x H x W
    #     src_masks = src_masks[:, None]
    #     target_masks = target_masks[:, None]

    #     with torch.no_grad():
    #         # sample point_coords
    #         point_coords = get_uncertain_point_coords_with_randomness(
    #             src_masks,
    #             lambda logits: calculate_uncertainty(logits),
    #             self.num_points,
    #             self.oversample_ratio,
    #             self.importance_sample_ratio,
    #         )
    #         # get gt labels
    #         target_masks = point_sample(
    #             target_masks,
    #             point_coords,
    #             align_corners=False,
    #         ).squeeze(1)

    #     src_masks = point_sample(
    #         src_masks,
    #         point_coords,
    #         align_corners=False,
    #     ).squeeze(1)

    #     if loss_type == "focal":
    #         loss_mask = sigmoid_focal_loss(src_masks, target_masks, num_masks)
    #         loss_name = f"loss_focal_{name}"
    #     elif loss_type == "bce":
    #         loss_mask = sigmoid_ce_loss_jit(src_masks, target_masks, num_masks)
    #         loss_name = f"loss_bce_{name}"

    #     losses = {
    #         loss_name: loss_mask,
    #         f"loss_dice_{name}": dice_loss_jit(src_masks, target_masks, num_masks),
    #     }

    #     del src_masks
    #     del target_masks

    #     return losses


    def _loss_masks(self, outputs, targets, indices, num_masks, name, loss_type="bce"):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert f"pred_{name}" in outputs, f"{name} not found in losses."

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs[f"pred_{name}"]
        src_masks = src_masks[src_idx]

        masks = [t[f"{name}"] for t in targets]
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

        del src_masks
        del target_masks

        return losses


    def loss_masks(self, outputs, targets, indices, num_masks, **kwargs):
        return self._loss_masks(outputs, targets, indices, num_masks, name="instance_masks")
    
    def loss_occluders(self, outputs, targets, indices, num_masks, **kwargs):
        return self._loss_masks(outputs, targets, indices, num_masks, name="occluder_masks")
    
    def loss_overlaps(self, outputs, targets, indices, num_masks, **kwargs):
        return self._loss_masks(outputs, targets, indices, num_masks, name="overlap_masks")
    
    def loss_visible(self, outputs, targets, indices, num_masks, **kwargs):
        return self._loss_masks(outputs, targets, indices, num_masks, name="visible_masks")
    
    def loss_activations(self, outputs, targets, indices, num_masks, **kwargs):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_iams"]["instance_iams"]
        src_masks = src_masks[src_idx]

        masks = [t['instance_masks'] for t in targets]
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
            "loss_bce_iams": sigmoid_ce_loss_jit(src_masks, target_masks, num_masks),
            "loss_dice_iams": dice_loss_jit(src_masks, target_masks, num_masks),
        }

        del src_masks
        del target_masks

        return losses
    

    def loss_boxes(self, outputs, targets, indices, num_boxes, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_bboxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_bboxes'][idx]
        tgt_boxes = torch.cat([t['bboxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(tgt_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    
    
    def get_loss(self, loss, outputs, targets, indices, num_masks, **kwargs):
        loss_map = {
            "labels": self.loss_labels_ce if self.semantic_ce_loss else self.loss_labels,
            "masks": self.loss_masks,
            "bboxes": self.loss_boxes,
            "occluders": self.loss_occluders,
            "overlaps": self.loss_overlaps,
            "visible": self.loss_visible,
            "iou": self.loss_iou,
            "activations": self.loss_activations,
        }
        
        assert loss in loss_map, f"loss {loss} not found in loss_map!"
        return loss_map[loss](outputs, targets, indices, num_masks, **kwargs)

    
    def forward(self, outputs, targets, return_matches=False, **kwargs):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target masks accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices,
                                        num_masks, **kwargs))
        
        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]


        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, 
                                           num_masks, **kwargs)
                    
                    for k in l_dict.keys():
                        if k in self.weight_dict:
                            l_dict[k] *= self.weight_dict[k]
                        
                    l_dict = {k + f'.{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)


        if return_matches:
            src_idx = self._get_src_permutation_idx(indices)
            tgt_idx = self._get_tgt_permutation_idx(indices)
            return losses, (src_idx, tgt_idx)

        return losses

    def __repr__(self, _repr_indent=4):
        head = "Criterion " + self.__class__.__name__

        body = [
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
            "semantic_ce_loss: {}".format(self.semantic_ce_loss),
            "matcher: \n{}".format(self.matcher),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]

        return "\n" + "\n".join(lines) + "\n"
    

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