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
    

    def loss_labels(self, outputs, targets, indices, num_instances, input_shape, **kwargs):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )  # [1, 1, 1, 1, 1, 1, ...] size(50)
        target_classes[idx] = target_classes_o # [1, 0, 1, 1, 0, 1, ...] size(50), where 0 is the obj class

        self.empty_weight = self.empty_weight.to(src_logits.device)
        # print(self.empty_weight.shape, self.empty_weight)

        # scores, labels = F.softmax(src_logits, dim=-1).max(-1)
        # print(f"ls: {scores.shape}")
        # print(scores.view(-1))
        # print(target_classes.view(-1))
        # print(src_logits.transpose(1, 2).shape, target_classes.shape)

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses
    


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



    # def loss_labels(self, outputs, targets, indices, num_instances, input_shape, **kwargs):
    #     assert "pred_logits" in outputs
    #     src_logits = outputs['pred_logits']
    #     idx = self._get_src_permutation_idx(indices)
    #     target_classes_o = torch.cat([t["labels"][J]
    #                                  for t, (_, J) in zip(targets, indices)])  # [cls1, cls2, cls1, cls1, ...] | clsi < num_classes
    #     target_classes = torch.full(src_logits.shape[:2], self.num_classes,
    #                                 dtype=torch.int64, device=src_logits.device)    # [1, 1, 1, 1, 1, ...] | bg cls
    #     target_classes[idx] = target_classes_o  # [1, cls2, 1, 1, cls1, cls1, 1, 1] # gt matching, everything else is bg == 1

    #     src_logits = src_logits.flatten(0, 1)
    #     # prepare one_hot target.
    #     target_classes = target_classes.flatten(0, 1)
    #     pos_inds = torch.nonzero(
    #         target_classes != self.num_classes, as_tuple=True)[0] # [1, cls2, 1, 1, cls1, cls1, 1, 1] -> [F, T, F, F, T, T, F, F] 
    #                                                               #  -> [1, 4, 5] - idx of cls that are not bg
    #     labels = torch.zeros_like(src_logits) # [0, 0, 0, 0, 0, 0, 0, ...]
    #     labels[pos_inds, target_classes[pos_inds]] = 1  # [0, _, 0, 0, _, _, 0, 0, ...] -> [0, 1, 0, 0, 1, 1, 0, 0, ...] - put ones where cls != bg

    #     print(src_logits.sigmoid().view(-1))
    #     print(labels.view(-1))
        
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


    def loss_iou(self, outputs, targets, indices, num_instances, input_shape, **kwargs):
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

        tgt_iou_scores = torch.full(
            src_iou_scores.shape[:2], 0, dtype=torch.float32, device=src_iou_scores.device
        )
        tgt_iou_scores[src_idx] = ious

        src_iou_scores = src_iou_scores.flatten(0)
        tgt_iou_scores = tgt_iou_scores.flatten(0)

        # print()
        # print(src_iou_scores.sigmoid())
        # print()
        # print(tgt_iou_scores)

        # iou_loss = sigmoid_focal_loss_jit(
        #     src_iou_scores,
        #     tgt_iou_scores,
        #     alpha=0.25,
        #     gamma=2.0,
        #     reduction="sum",
        # ) / num_instances

        iou_loss = F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean')

        losses = {"loss_objectness_masks": iou_loss}
        return losses
    
    

    def loss_masks(self, outputs, targets, indices, num_instances, input_shape, epoch):
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
            "loss_bce_masks": F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean'),
            "loss_dice_masks": dice_loss_detr(src_masks, target_masks, num_instances),
        }
        return losses
    
    
    def get_loss(self, loss, outputs, targets, indices, num_instances, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
            "iou": self.loss_iou,
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



# class MILInstanceBranch(nn.Module):
#     def __init__(self, dim, num_masks=10, num_classes=1):
#         super().__init__()

#         self.num_classes = num_classes

#         # iam prediction, a simple conv
#         self.iam_conv = nn.Conv2d(dim, num_masks, 3, padding=1)

#         self.slice_attention_conv = nn.Conv2d(dim, num_masks * self.num_classes, 3, padding=1)
#         self.iam_proccessing_fc = nn.Linear(dim, dim)
#         self.fc = nn.Linear(dim, dim)

#         # outputs
#         self.cls_score = nn.Linear(dim, self.num_classes)
#         self.attention_score = nn.Linear(dim, 1)

#         self.prior_prob = 0.01
#         self._init_weights()
#         self.softmax_bias = nn.Parameter(torch.ones([1, ]))

#     def _init_weights(self):
#         bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
#         for module in [self.iam_conv, self.cls_score]:
#             init.constant_(module.bias, bias_value)
#         init.normal_(self.iam_conv.weight, std=0.01)
#         init.normal_(self.cls_score.weight, std=0.01)
#         c2_xavier_fill(self.fc)

#     def forward(self, features, idx=None):
#         # predict instance activation maps
#         iam = self.iam_conv(features)

#         B, N, H, W = iam.shape
#         C = features.size(1)

#         # BxNxHxW -> BxNx(HW)
#         iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)

#         # aggregate features: BxCxHxW -> Bx(HW)xC
#         inst_features = torch.bmm(
#             iam_prob, features.view(B, C, -1).permute(0, 2, 1))

#         inst_features = inst_features.reshape(
#             B, 1, N, -1).transpose(1, 2).reshape(B, N, -1)

#         slice_attention = F.relu_(self.fc(F.softmax(inst_features, dim=0)))

#         slice_scores = self.attention_score(slice_attention)
#         slice_scores = F.softmax(slice_scores, dim=0)  # softmax over N

#         slice_scores = torch.squeeze(slice_scores, 1).T
#         inst_features = torch.squeeze(inst_features)

#         M = torch.mm(slice_scores, inst_features)  # KxL

#         Y_prob = self.cls_score(M)
#         # print("iam scores: ",Y_prob)
#         iam = {
#             "iam": iam,
#         }

#         Y_hat = nn.Sigmoid()(Y_prob)
#         Y_hat = torch.ge(Y_hat, 0.5).float()
#         # print("iam yhat: ", Y_hat)
#         return Y_prob, Y_hat, slice_scores, iam