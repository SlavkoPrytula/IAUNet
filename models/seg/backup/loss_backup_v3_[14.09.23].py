import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.utils import nested_masks_from_list, compute_mask_iou
from utils.comm import is_dist_avail_and_initialized, get_world_size
from utils.losses import dice_loss, sigmoid_focal_loss_jit

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


    def get_weight_dict(self):
        losses = ("loss_ce", "loss_bce_masks", "loss_dice_masks", "loss_objectness_masks", 
                  "loss_bce_occluders", "loss_dice_occluders", "loss_objectness_occluders")
        weight_dict = {}

        ce_weight = self.loss_weights.labels
        
        # mask.
        bce_masks_weight = self.loss_weights.bce_masks
        dice_masks_weight = self.loss_weights.dice_masks
        objectness_masks_weight = self.loss_weights.iou_masks
        
        # occluders.
        bce_occluders_weight = 5.0
        dice_occluders_weight = 2.0
        objectness_occluders_weight = 1.0


        weight_dict = dict(
            zip(losses, (ce_weight, bce_masks_weight, dice_masks_weight, objectness_masks_weight, 
                         bce_occluders_weight, dice_occluders_weight, objectness_occluders_weight)))
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


    # def loss_labels(self, outputs, targets, indices, num_instances, input_shape=None):
    #     assert "pred_logits" in outputs
    #     src_logits = outputs['pred_logits']

    #     idx = self._get_src_permutation_idx(indices)
    #     # idx = self._get_src_permutation_idx(indices)
    #     # src_idx = self._get_src_permutation_idx(indices)
    #     # tgt_idx = self._get_tgt_permutation_idx(indices)

    #     target_classes_o = torch.cat([t["labels"][J]
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
    #     losses = {'loss_ce': class_loss}
    #     return losses


    # def loss_labels(self, outputs, targets, indices, num_instances, input_shape, **kwargs):
    #     """Classification loss (NLL)
    #     targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    #     """
    #     assert "pred_logits" in outputs
    #     src_logits = outputs["pred_logits"]

    #     idx = self._get_src_permutation_idx(indices)
    #     target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
    #     target_classes = torch.full(
    #         src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
    #     )
    #     target_classes[idx] = target_classes_o

    #     # num_classes_p1 = src_logits.size(2)
    #     # with torch.no_grad():
    #     #     loss_weights = torch.ones(num_classes_p1, dtype=torch.float32, device=src_logits.device, requires_grad=False)
    #     #     loss_weights[0] = 0.1
    #     # empty_weight = self.empty_weight.to(src_logits.device)
        
    #     # target_classes = target_classes.unsqueeze(1)
    #     target_classes = target_classes.unsqueeze(-1)

    #     print(src_logits.shape, target_classes.shape) # (2, 50, 1) (2, 1, 50)
    #     print(src_logits.transpose(1, 2).shape) # (2, 1, 50)
    #     # print(loss_weights)

    #     loss_ce = F.cross_entropy(src_logits.float(), target_classes.float())
    #     losses = {"loss_ce": loss_ce}

    #     return losses


    def loss_labels(self, outputs, targets, indices, num_instances, input_shape=None):
        assert "pred_logits" in outputs
        src_logits = outputs['pred_logits']

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        # (B, N, 1)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        

        # (B, N, 1), map labels to matched predictions
        src_logits = src_logits[src_idx]
        target_classes[tgt_idx] = target_classes_o


        # (B, N, 1) -> (N)
        src_logits = src_logits.flatten(0, 1)
        

        # (B, N, 1) -> (N)
        # [0, 1, 1, ..., 0, 1, ...]
        target_classes = target_classes.flatten(0, 1)
        
        # TODO: check this (should be pos_inds = target_classes[target_classes == 0])
        # get positions of zero values (!= num_classes)
        pos_inds = torch.nonzero(target_classes != self.num_classes).squeeze(1)

        # prepare one_hot target.
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


        src_iams = outputs["pred_iam"]["iam"]
        src_iams = src_iams[src_idx]

        vis_preds_cyto = src_masks.sigmoid().cpu().detach().numpy()
        # vis_preds_iams = src_iams.sigmoid().cpu().detach().numpy()
        N, H, W = src_iams.shape
        vis_preds_iams = F.softmax(src_iams.view(N, -1), dim=-1).view(N, H, W).cpu().detach().numpy()
        vis_gt_cyto = target_masks.cpu().detach().numpy()

        visualize_grid_v2(
            masks=vis_preds_cyto, 
            ncols=5, 
            path=f'{self.cfg.save_dir}/valid_visuals/cyto_pred.jpg',
            cmap='jet'
        )
        
        visualize_grid_v2(
            masks=vis_preds_iams, 
            ncols=5, 
            path=f'{self.cfg.save_dir}/valid_visuals/iams_pred.jpg',
            cmap='jet'
        )

        visualize_grid_v2(
            masks=vis_gt_cyto, 
            ncols=5, 
            path=f'{self.cfg.save_dir}/valid_visuals/cyto_gt.jpg',
            cmap='jet'
        )


        
        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)

        src_iams = src_iams.flatten(1)


        # with torch.no_grad():
        #     ious = compute_mask_iou(src_masks, target_masks)
        # tgt_iou_scores = ious
        # src_iou_scores = src_iou_scores[src_idx]

        # tgt_iou_scores = tgt_iou_scores.flatten(0)
        # src_iou_scores = src_iou_scores.flatten(0)

        losses = {
            # "loss_objectness": F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean'),
            # "loss_objectness_masks": nn.MSELoss()(src_iou_scores, tgt_iou_scores),
            "loss_dice_masks": dice_loss(src_masks, target_masks) / num_instances,
            "loss_bce_masks": F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean') # + F.binary_cross_entropy_with_logits(src_iams, target_masks, reduction='mean')
        }
        
        return losses 
    

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
    
    
    def get_loss(self, loss, outputs, targets, indices, num_instances, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks_with_iou_objectness,
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