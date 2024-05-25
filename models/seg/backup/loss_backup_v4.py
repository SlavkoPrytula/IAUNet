import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append("./")

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
        self.num_classes = cfg.num_classes # - 1
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
        focal_masks_weight = self.loss_weights.focal_masks
        dice_masks_weight = self.loss_weights.dice_masks
        bce_masks_weight = self.loss_weights.bce_masks
        objectness_masks_weight = self.loss_weights.iou_masks
        
        # occluders.
        ce_occluders_weight = 2.0
        focal_occluders_weight = 5.0
        bce_occluders_weight = 5.0
        dice_occluders_weight = 2.0
        objectness_occluders_weight = 1.0


        weight_dict = dict(
            zip(losses, (ce_weight, focal_masks_weight, bce_masks_weight, dice_masks_weight, objectness_masks_weight, 
                         ce_occluders_weight, focal_occluders_weight, bce_occluders_weight, dice_occluders_weight, objectness_occluders_weight)))
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

    #     self.empty_weight = self.empty_weight.to(src_logits.device)

    #     # print(src_logits.shape) # (2, 50, 1)
    #     # print(src_logits.transpose(1, 2).shape, target_classes.shape) # (2, 1, 50), (2, 50)
    #     # print(self.empty_weight.shape) # (2)

    #     print(src_logits)
    #     print(target_classes)
    #     # raise


    #     # loss_ce = nn.CrossEntropyLoss(
    #     #     src_logits.transpose(1, 2).float(), 
    #     #     target_classes.float(), 
    #     #     reduction='mean'
    #     #     )
    #     loss_ce = F.cross_entropy(src_logits.transpose(1, 2).float(), target_classes) #, self.empty_weight)
    #     # print(loss_ce)
    #     losses = {"loss_ce": loss_ce}
    #     return losses
    

    # def loss_labels(self, outputs, targets, indices, num_instances, input_shape, **kwargs):
    #     """
    #     Classification loss (NLL)
    #     targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    #     """
    #     assert 'pred_logits' in outputs
    #     src_logits = outputs['pred_logits']

    #     idx = self._get_src_permutation_idx(indices)
    #     target_classes_o = torch.cat(
    #         [t["labels"][J] for t, (_, J) in zip(targets, indices)]
    #     )
    #     target_classes = torch.full(
    #         src_logits.shape[:2],
    #         self.num_classes,
    #         dtype=torch.int64,
    #         device=src_logits.device,
    #     )
    #     target_classes[idx] = target_classes_o

    #     target_classes_onehot = torch.zeros(
    #         [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
    #         dtype=src_logits.dtype,
    #         layout=src_logits.layout,
    #         device=src_logits.device,
    #     )
    #     target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
    #     target_classes_onehot = target_classes_onehot[:, :, :-1]

    #     loss_ce = (
    #         sigmoid_focal_loss_hdetr(
    #             src_logits, #.squeeze(-1),
    #             target_classes_onehot, #.squeeze(-1),
    #             num_instances,
    #             alpha=0.25,
    #             gamma=2,
    #         )
    #         * src_logits.shape[1]
    #     )
    #     losses = {"loss_ce": loss_ce}

    # #     # if log:
    # #     #     # TODO this should probably be a separate loss, not hacked in this one here
    # #     #     losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
    #     return losses



    def loss_labels(self, outputs, targets, indices, num_instances, input_shape, **kwargs):
        assert "pred_logits" in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J]
                                     for t, (_, J) in zip(targets, indices)])  # [cls1, cls2, cls1, cls1, ...] | clsi < num_classes
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)    # [1, 1, 1, 1, 1, ...] | bg cls
        target_classes[idx] = target_classes_o  # [1, cls2, 1, 1, cls1, cls1, 1, 1] # gt matching, everything else is bg == 1

        src_logits = src_logits.flatten(0, 1)
        # prepare one_hot target.
        target_classes = target_classes.flatten(0, 1)
        pos_inds = torch.nonzero(
            target_classes != self.num_classes, as_tuple=True)[0] # [1, cls2, 1, 1, cls1, cls1, 1, 1] -> [F, T, F, F, T, T, F, F] 
                                                                  #  -> [1, 4, 5] - idx of cls that are not bg
        labels = torch.zeros_like(src_logits) # [0, 0, 0, 0, 0, 0, 0, ...]
        labels[pos_inds, target_classes[pos_inds]] = 1  # [0, _, 0, 0, _, _, 0, 0, ...] -> [0, 1, 0, 0, 1, 1, 0, 0, ...] - put ones where cls != bg

        print(src_logits.sigmoid().view(-1))
        print(labels.view(-1))
        
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
    

    def center_of_mass(self, bitmasks):
        _, h, w = bitmasks.size()
        ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
        xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

        m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
        m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
        m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
        center_x = m10 / m00
        center_y = m01 / m00
        return center_x, center_y
    

    def loss_masks(self, outputs, targets, indices, num_instances, input_shape, epoch):
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


        # src_coords = outputs["pred_coords"]  # (B, N, 2)
        # src_coords = src_coords[src_idx]

        src_masks = src_masks.squeeze(1)
        target_masks = target_masks.squeeze(1)

        # x, y = self.center_of_mass(target_masks)
        # target_coords = torch.stack([x / target_masks.shape[-2], y / target_masks.shape[-1]]).t()

        # a = src_coords.sigmoid().cpu().detach().numpy()
        # b = target_coords.cpu().detach().numpy()

        # for i, j in zip(a, b):
        #     print(i, j)
        # raise


        # src_iams = outputs["pred_iam"]
        # src_iams = src_iams[src_idx]

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
        # src_iams = src_iams.flatten(1)

        _loss = F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean')
        # if epoch < 400: 
        #     _loss += F.binary_cross_entropy_with_logits(src_iams, target_masks, reduction='mean')

        losses = {
            # "loss_focal_masks": sigmoid_focal_loss_hdetr(src_masks, target_masks, num_instances),
            # "loss_bce_iams": F.binary_cross_entropy_with_logits(src_iams, target_masks, reduction='mean'),
            # "loss_centerness_masks": F.binary_cross_entropy_with_logits(src_coords, target_coords, reduction='mean'),
            "loss_bce_masks": _loss, # F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean'), # + F.binary_cross_entropy_with_logits(src_iams, target_masks, reduction='mean'),
            "loss_dice_masks": dice_loss_detr(src_masks, target_masks, num_instances),
        }
        return losses
    


    def _loss_labels_occluders(self, outputs, targets, indices, num_instances, input_shape):
        """
        Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_occluders_logits' in outputs
        src_logits = outputs['pred_occluders_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels_occluders"][J] for t, (_, J) in zip(targets, indices)]
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
        losses = {"loss_ce_occluders": loss_ce}
        return losses


    def _loss_mask_occluders(self, outputs, targets, indices, num_instances, input_shape, name):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert f"pred_{name}_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs[f"pred_{name}_masks"]
        src_masks = src_masks[src_idx]

        masks = [t[name] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = F.interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                    mode="bilinear", align_corners=False)
        

        src_coords = outputs[f"pred_{name}_coords"]  # (B, N, 2)
        src_coords = src_coords[src_idx]

        src_masks = src_masks.squeeze(1)
        target_masks = target_masks.squeeze(1)

        x, y = self.center_of_mass(target_masks)
        target_coords = torch.stack([x / target_masks.shape[-2], y / target_masks.shape[-1]]).t()


        src_masks = src_masks.squeeze(1)
        target_masks = target_masks.squeeze(1)

        # src_iams = outputs["pred_iam"][f"occluder_iam"]
        # src_iams = src_iams[src_idx]

        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)
        # src_iams = src_iams.flatten(1)

        losses = {
            # f"loss_focal_{name}": sigmoid_focal_loss_hdetr(src_masks, target_masks, num_instances),
            f"loss_centerness_{name}": F.binary_cross_entropy_with_logits(src_coords, target_coords, reduction='mean'),
            f"loss_bce_{name}": F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean'), # + F.binary_cross_entropy_with_logits(src_iams, target_masks, reduction='mean'),
            f"loss_dice_{name}": dice_loss_detr(src_masks, target_masks, num_instances),
        }
        return losses   


    def loss_occluders(self, outputs, targets, indices, num_instances, input_shape, **kwargs):
        losses = {}
        for loss_name in ["occluders"]:
            loss = self._loss_mask_occluders(outputs, targets, indices, num_instances, input_shape, name=loss_name)
            losses.update(loss)

        # loss = self._loss_labels_occluders(outputs, targets, indices, num_instances, input_shape)
        # losses.update(loss)

        return losses
    
    
    def get_loss(self, loss, outputs, targets, indices, num_instances, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
            "occluders": self.loss_occluders,
        }
        assert loss in loss_map, f"loss {loss} not found in loss_map!"
        return loss_map[loss](outputs, targets, indices, num_instances, **kwargs)

    
    def forward(self, outputs, targets, input_shape, return_matches=False, **kwargs):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, input_shape)
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
                                        num_instances, input_shape=input_shape, **kwargs))
        
        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]


        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # indices = self.matcher(aux_outputs, targets, input_shape)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, 
                                           num_instances, input_shape=input_shape, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)


        if return_matches:
            src_idx = self._get_src_permutation_idx(indices)
            tgt_idx = self._get_tgt_permutation_idx(indices)
            return losses, (src_idx, tgt_idx)

        return losses



# if __name__ == "__main__":
#     # import sys
#     # sys.path.append("./")

#     from models.seg.matcher import HungarianMatcher
#     from utils.registry import MATCHERS
#     print(MATCHERS)

#     criterion = CRITERIONS.build(cfg.model.criterion)
#     print(criterion)
