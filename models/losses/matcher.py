import torch 
from torch import nn
import torch.nn.functional as F
from torch.cuda import amp

import sys
sys.path.append("./")

from scipy.optimize import linear_sum_assignment
from utils.utils import nested_tensor_from_tensor_list
from utils.losses import batch_sigmoid_ce_loss_jit, batch_dice_loss_jit
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from configs import cfg
from utils.registry import MATCHERS


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output



        
@MATCHERS.register(name="HungarianMatcher")
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cfg: cfg):
        """Creates the matcher

        Params:
            cfg: This is a dict type sub-config containing the params from creating the matcher 
            matcher = registy.get("HungarianMatcher")(cfg.model.criterion.matcher)

            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cfg.cost_cls
        self.cost_mask = cfg.cost_mask
        self.cost_dice = cfg.cost_dice
        self.cost_bbox = cfg.cost_bbox
        self.cost_giou = cfg.cost_giou
        # assert 1 != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            pred_mask = outputs["pred_instance_masks"][b]  # [num_queries, H, W]

            tgt_ids = targets[b]["labels"]
            tgt_mask = targets[b]["instance_masks"].to(pred_mask)

            # Downsample gt masks to save memory
            tgt_mask = F.interpolate(tgt_mask[:, None], size=pred_mask.shape[-2:], mode="nearest")

            # Flatten spatial dimension
            pred_mask = pred_mask.flatten(1)  # [batch_size * num_queries, H*W]
            tgt_mask = tgt_mask[:, 0].flatten(1)  # [num_total_targets, H*W]


            with amp.autocast(enabled=False):
                pred_mask = pred_mask.float()
                tgt_mask = tgt_mask.float()
                out_prob = out_prob.float()

                # v1
                # Compute the classification cost. Contrary to the loss, we don't use the NLL,
                # but approximate it in 1 - proba[target class].
                # The 1 is a constant that doesn't change the matching, it can be ommitted.
                cost_class = -out_prob[:, tgt_ids]

                # v2 - same as focal loss
                # Compute the classification cost.
                # alpha = 0.25
                # gamma = 2.0
                # neg_cost_class = (
                #     (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
                # )
                # pos_cost_class = (
                #     alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
                # )
                # cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]


                # Compute the focal loss between masks
                # cost_mask = batch_sigmoid_focal_loss(pred_mask, tgt_mask)
                # cost_iams_mask = batch_sigmoid_focal_loss(out_iams, tgt_mask)

                # binary cross entropy cost
                cost_mask = batch_sigmoid_ce_loss_jit(pred_mask, tgt_mask)
                
                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(pred_mask, tgt_mask)

                # Final cost matrix
                C = (
                    self.cost_class * cost_class
                    + self.cost_dice * cost_dice
                    + self.cost_mask * cost_mask
                )
            
            C = C.reshape(num_queries, -1).cpu()
            C = torch.nan_to_num(C, nan=0)
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)




@MATCHERS.register(name="PointSampleHungarianMatcher")
class PointSampleHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cfg: cfg):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cfg.cost_cls
        self.cost_mask = cfg.cost_mask
        self.cost_dice = cfg.cost_dice
        self.cost_bbox = cfg.cost_bbox
        self.cost_giou = cfg.cost_giou

        self.num_points = 112 * 112
        print(self)


    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []

        # Iterate through batch size
        for b in range(bs):
            
            # cls.
            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            # v2 - same as focal loss
            # Compute the classification cost.
            # alpha = 0.25
            # gamma = 2.0
            # neg_cost_class = (
            #     (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            # )
            # pos_cost_class = (
            #     alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            # )
            # cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]


            # masks.
            pred_mask = outputs["pred_instance_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["instance_masks"].to(pred_mask)

            pred_mask = pred_mask[:, None]
            tgt_mask = tgt_mask[:, None]

            # tgt_mask = F.interpolate(tgt_mask, size=pred_mask.shape[-2:], mode="nearest")

            # bboxes.
            # out_bbox = outputs["pred_bboxes"][b]
            # tgt_bbox = targets[b]["bboxes"].to(out_bbox)

            # cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            # cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))


            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=pred_mask.device)
            
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            pred_mask = point_sample(
                pred_mask,
                point_coords.repeat(pred_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)
            # tgt_mask = tgt_mask.flatten(1)
            # pred_mask = pred_mask.flatten(1)

            # iams.
            # pred_iams = outputs["pred_iams"]["instance_iams"][b]
            # pred_iams = point_sample(
            #     tgt_mask.unsqueeze(0),
            #     pred_iams.unsqueeze(0),
            #     align_corners=False,
            # ).squeeze(0)
            # cost_iams = (cost_iams > 0).to(pred_mask)
            # cost_iams = -cost_iams.transpose(0, 1)

            with amp.autocast(enabled=False):
                pred_mask = pred_mask.float()
                tgt_mask = tgt_mask.float()

                # Compute the bce loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(pred_mask, tgt_mask)
                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(pred_mask, tgt_mask)
            
            C = (
                self.cost_class * cost_class
                + self.cost_mask * cost_mask
                + self.cost_dice * cost_dice
                # + self.cost_bbox * cost_bbox 
                # + self.cost_giou * cost_giou
                # + self.cost_mask * cost_iams
            )
            C = C.reshape(num_queries, -1).cpu()
            # quick fix.
            C = torch.nan_to_num(C, nan=0)
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
            "cost_bbox: {}".format(self.cost_bbox),
            "cost_giou: {}".format(self.cost_giou),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n" + "\n".join(lines) + "\n"

