import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.utils import nested_masks_from_list, nested_tensor_from_tensor_list, compute_mask_iou
from utils.comm import is_dist_avail_and_initialized, get_world_size
from utils.losses import dice_loss, sigmoid_focal_loss_jit, sigmoid_focal_loss_hdetr, dice_loss_detr

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
                 "loss_focal_occluders", "loss_bce_occluders", "loss_dice_occluders", "loss_objectness_occluders")
        weight_dict = {}

        ce_weight = self.loss_weights.labels
        
        # mask.
        focal_masks_weight = self.loss_weights.focal_masks
        dice_masks_weight = self.loss_weights.dice_masks
        bce_masks_weight = self.loss_weights.bce_masks
        objectness_masks_weight = self.loss_weights.iou_masks
        
        # occluders.
        focal_occluders_weight = 5.0
        bce_occluders_weight = 5.0
        dice_occluders_weight = 2.0
        objectness_occluders_weight = 1.0


        weight_dict = dict(
            zip(losses, (ce_weight, focal_masks_weight, bce_masks_weight, dice_masks_weight, objectness_masks_weight, 
                         focal_occluders_weight, bce_occluders_weight, dice_occluders_weight, objectness_occluders_weight)))
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
    

    # def loss_labels(self, outputs, targets, indices, num_instances, input_shape):
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
        
    #     target_classes = target_classes.unsqueeze(1)

    #     print(src_logits.shape, target_classes.shape)
    #     print(src_logits.transpose(1, 2).shape)
    #     # print(loss_weights)
    #     loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
    #     losses = {"loss_ce": loss_ce}
    #     print(losses)
    #     raise
    #     return losses

    def loss_labels(self, outputs, targets, indices, num_instances, input_shape):
        """
        Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = (
            sigmoid_focal_loss_hdetr(
                src_logits, #.squeeze(-1),
                target_classes_onehot, #.squeeze(-1),
                num_instances,
                alpha=0.25,
                gamma=2,
            )
            * src_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

    #     # if log:
    #     #     # TODO this should probably be a separate loss, not hacked in this one here
    #     #     losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_masks(self, outputs, targets, indices, num_instances, input_shape):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

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


        src_iams = outputs["pred_iam"]["iam"]
        src_iams = src_iams[src_idx]

        # vis_preds_cyto = src_masks.sigmoid().cpu().detach().numpy()
        
        # vis_preds_iams_sigmoid = src_iams.sigmoid().cpu().detach().numpy()
        
        # N, H, W = src_iams.shape
        # vis_preds_iams_softmax = F.softmax(src_iams.view(N, -1), dim=-1).view(N, H, W).cpu().detach().numpy()
        # vis_gt_cyto = target_masks.cpu().detach().numpy()

        # visualize_grid_v2(
        #     masks=vis_preds_cyto, 
        #     ncols=5, 
        #     path=f'{self.cfg.save_dir}/valid_visuals/cyto_pred.jpg',
        #     cmap='jet'
        # )
        
        # visualize_grid_v2(
        #     masks=vis_preds_iams_softmax, 
        #     ncols=5, 
        #     path=f'{self.cfg.save_dir}/valid_visuals/iams_pred_softmax.jpg',
        #     cmap='jet'
        # )

        # visualize_grid_v2(
        #     masks=vis_preds_iams_sigmoid, 
        #     ncols=5, 
        #     path=f'{self.cfg.save_dir}/valid_visuals/iams_pred_sigmoid.jpg',
        #     cmap='jet'
        # )

        # visualize_grid_v2(
        #     masks=vis_gt_cyto, 
        #     ncols=5, 
        #     path=f'{self.cfg.save_dir}/valid_visuals/cyto_gt.jpg',
        #     cmap='jet'
        # )

        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)
        src_iams = src_iams.flatten(1)

        losses = {
            # "loss_focal_masks": sigmoid_focal_loss_hdetr(src_masks, target_masks, num_instances),
            "loss_bce_masks": F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean'), #+ F.binary_cross_entropy_with_logits(src_iams, target_masks, reduction='mean'),
            "loss_dice_masks": dice_loss_detr(src_masks, target_masks, num_instances),
        }
        return losses
    

    def _loss_mask_occluders(self, outputs, targets, indices, num_instances, input_shape, name):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert f"pred_{name}" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs[f"pred_{name}"]
        src_masks = src_masks[src_idx]

        masks = [t[name] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = F.interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                    mode="bilinear", align_corners=False)

        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)

        losses = {
            f"loss_focal_{name}": sigmoid_focal_loss_hdetr(src_masks, target_masks, num_instances),
            f"loss_bce_{name}": F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean'),
            f"loss_dice_{name}": dice_loss_detr(src_masks, target_masks, num_instances),
        }
        return losses   


    def loss_occluders(self, outputs, targets, indices, num_instances, input_shape):
        losses = {}
        for loss_name in ["occluders"]:
            loss = self._loss_mask_occluders(outputs, targets, indices, num_instances, input_shape, name=loss_name)
            losses.update(loss)

        return losses
    
    
    def get_loss(self, loss, outputs, targets, indices, num_instances, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
            "occluders": self.loss_occluders,
        }
        assert loss in loss_map
        return loss_map[loss](outputs, targets, indices, num_instances, **kwargs)

    
    def forward(self, outputs, targets, input_shape, return_matches=False):
        # outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets, input_shape)
        # indices = self.matcher(outputs, targets)

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
