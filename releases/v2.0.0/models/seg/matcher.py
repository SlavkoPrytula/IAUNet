import torch 
from torch import nn
import torch.nn.functional as F
from torch.cuda import amp

import sys
sys.path.append("./")

from scipy.optimize import linear_sum_assignment
from utils.utils import nested_masks_from_list, nested_tensor_from_tensor_list
from utils.losses import dice_score

from configs import cfg
from utils.registry import MATCHERS


def batch_dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum(
        "nc,mc->nm", focal_neg, (1 - targets)
    )

    return loss / hw



# @MATCHERS.register(name="HungarianMatcher")
# class HungarianMatcher(nn.Module):
#     def __init__(self, cfg: cfg):
#         super().__init__()
#         # self.alpha = cfg.model.criterion.matcher.mask_cost
#         # self.beta = cfg.model.criterion.matcher.cls_cost
        
#         self.mask_cost = cfg.cost_mask
#         self.cls_cost = cfg.cost_cls
        
#         self.mask_score = dice_score

#     def forward(self, outputs, targets, input_shape):
#         with torch.no_grad():
#             B, N, H, W = outputs["pred_masks"].shape

#             pred_masks = outputs['pred_masks']
#             pred_logits = outputs['pred_logits'].flatten(0, 1).sigmoid()
#             # pred_logits = outputs['pred_logits'].sigmoid()
#             # print(pred_logits.shape)
            
#             tgt_ids = torch.cat([v["labels"] for v in targets])

#             if tgt_ids.shape[0] == 0:
#                 return [(torch.as_tensor([]).to(pred_logits), torch.as_tensor([]).to(pred_logits))] * B
            
#             masks = [t["masks"] for t in targets]
#             # target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
#             # target_masks = target_masks.to(pred_masks)
            
#             target_masks, valid = nested_masks_from_list(masks, input_shape).decompose()
#             target_masks = target_masks.to(pred_masks)

#             # tgt_masks = F.interpolate(
#             #     tgt_masks[:, None], size=pred_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

#             pred_masks = pred_masks.view(B * N, -1)
#             target_masks = target_masks.flatten(1)
#             # print(pred_masks.shape, target_masks.shape)
#             # pred_masks = pred_masks.flatten(1)
#             # target_masks = target_masks.flatten(1)


#             # Compute the classification cost.
#             # alpha = 0.25
#             # gamma = 2.0
#             # neg_cost_class = (
#             #     (1 - alpha) * (pred_logits ** gamma) * (-(1 - pred_logits + 1e-8).log())
#             # )
#             # pos_cost_class = (
#             #     alpha * ((1 - pred_logits) ** gamma) * (-(pred_logits + 1e-8).log())
#             # )
#             # cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]


#             with amp.autocast(enabled=False):
#                 pred_masks = pred_masks.float()
#                 target_masks = target_masks.float()
#                 pred_logits = pred_logits.float()

#                 mask_score = self.mask_score(pred_masks, target_masks)
#                 # print(mask_score.shape)
#                 # print(cost_class.shape)
#                 # print(outputs['pred_logits'][:, tgt_ids].shape)
                
#                 # Nx(Number of gts)
#                 cost_class = pred_logits.view(B * N, -1)[:, tgt_ids]
#                 # C = (mask_score ** self.mask_cost) * (matching_prob ** self.cls_cost)
#                 C = (mask_score * self.mask_cost) + (cost_class * self.cls_cost)

#             C = C.view(B, N, -1).cpu()
#             C = torch.nan_to_num(C, nan=0, posinf=0, neginf=0) # FIXME:

#             # hungarian matching
#             sizes = [len(v["masks"]) for v in targets]
#             # indices = [linear_sum_assignment(c[i], maximize=True)
#             #            for i, c in enumerate(C.split(sizes, -1))]
#             # indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
#             # return indices

#             indices = [
#                 linear_sum_assignment(c[i], maximize=True) for i, c in enumerate(C.split(sizes, -1))
#             ]
#             return [
#                 (
#                     torch.as_tensor(i, dtype=torch.int64),
#                     torch.as_tensor(j, dtype=torch.int64),
#                 )
#                 for i, j in indices
#             ]


# with torch.cuda.amp.autocast(False):
    #         # binary cross entropy cost
    #         pos_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.ones_like(out_mask), reduction='none')
    #         neg_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.zeros_like(out_mask), reduction='none')
    #         cost_mask = torch.matmul(pos_cost_mask, tgt_mask.T) + torch.matmul(neg_cost_mask, 1 - tgt_mask.T)
    #         cost_mask /= self.num_sample_points
        
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
        # assert 1 != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Work out the mask padding size
        # masks = [v["masks"] for v in targets]
        # h_max = max([m.shape[1] for m in masks])
        # w_max = max([m.shape[2] for m in masks])

        indices = []

        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            out_mask = outputs["pred_masks"][b]  # [num_queries, H, W]
            out_iams = outputs["pred_iam"]["iam"][b]  # [num_queries, H, W]

            tgt_ids = targets[b]["labels"]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            # Downsample gt masks to save memory
            tgt_mask = F.interpolate(tgt_mask[:, None], size=out_mask.shape[-2:], mode="nearest")

            # Flatten spatial dimension
            out_mask = out_mask.flatten(1)  # [batch_size * num_queries, H*W]
            out_iams = out_iams.flatten(1)  # [batch_size * num_queries, H*W]
            tgt_mask = tgt_mask[:, 0].flatten(1)  # [num_total_targets, H*W]


            with amp.autocast(enabled=False):
                out_mask = out_mask.float()
                out_iams = out_iams.float()
                tgt_mask = tgt_mask.float()
                out_prob = out_prob.float()

                # v1
                # Compute the classification cost. Contrary to the loss, we don't use the NLL,
                # but approximate it in 1 - proba[target class].
                # The 1 is a constant that doesn't change the matching, it can be ommitted.
                # cost_class = -out_prob[:, tgt_ids]

                # v2 - same as focal loss
                # Compute the classification cost.
                alpha = 0.25
                gamma = 2.0
                neg_cost_class = (
                    (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
                )
                pos_cost_class = (
                    alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
                )
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_focal_loss(out_mask, tgt_mask)
                cost_iams = batch_sigmoid_focal_loss(out_iams, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss(out_mask, tgt_mask)

                # Final cost matrix
                C = (
                    self.cost_mask * cost_mask
                    + 5 * cost_iams
                    + self.cost_class * cost_class
                    + self.cost_dice * cost_dice
                )
            
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets, _):
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




def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def _get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx


if __name__ == "__main__":
    from utils.visualise import visualize_grid_v2

    mask1 = torch.zeros(2, 5, 10, 10)
    mask1[0, 0, :5, :5] = 1
    mask1[0, 1, 6:8, 6:8] = 1
    mask1[0, 2, 6:8, 0:3] = 1

    mask1[1, 0, 7:, 7:] = 1
    mask1[1, 1, 6:8, 7:8] = 1
    mask1[1, 2, :1, :1] = 1
    mask1[1, 3, :8, :1] = 1


    mask2 = torch.zeros(2, 3, 10, 10)
    mask2[0, 0, :5, :5] = 1
    mask2[0, 1, 6:8, 7:8] = 1
    mask2[0, 2, 6:8, 0:3] = 1

    mask2[1, 0, :1, :1] = 1
    mask2[1, 2, 6:8, 6:8] = 1
    mask2[1, 1, :2, :1] = 1
    
    # mask3 = torch.zeros(1, 4, 10, 10)
    # mask3[0, 0, :5, :5] = 1
    # mask3[0, 1, 6:8, 7:8] = 1
    # mask3[0, 2, 6:8, 0:3] = 1

    # mask3[0, 0, :1, :1] = 1
    # mask3[0, 2, 6:8, 6:8] = 1
    # mask3[0, 1, :2, :1] = 1


    # labels1 = torch.zeros(2, 3, 1, dtype=torch.int64)
    outputs = {
        "pred_masks": mask1,
        "pred_logits": torch.tensor([[0.7, 0.6, 0.6, 0.1, 0.1], [0.3, 0.5, 0.6, 0.1, 0.1]]).unsqueeze(-1)
    }

    labels1 = torch.zeros(3, dtype=torch.int64)
    labels2 = torch.zeros(3, dtype=torch.int64)
    targets = [
        {
            "masks": mask2[0], 
            "labels": labels1
        },
        {
            "masks": mask2[1], 
            "labels": labels2
        }
    ]


    num_classes = 2
    matcher = HungarianMatcher(cfg=cfg.model.criterion.matcher)
    indices = matcher(outputs, targets, [10, 10])
    # indices = matcher(outputs, targets)
    print(indices)

    # criterion = SparseInstCriterion(cfg.model.criterion, matcher=matcher)
    # loss = criterion(outputs, targets, [10, 10])
    # print(loss)

    # raise


    src_idx = _get_src_permutation_idx(indices)
    tgt_idx = _get_tgt_permutation_idx(indices)

    src_logits = outputs['pred_logits']

    print(src_idx)
    print(tgt_idx)
    print()

    target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
    
    # (B, N, 1)
    target_classes = torch.full(src_logits.shape[:2], num_classes,
                                dtype=torch.int64, device=src_logits.device)
    
    print(target_classes)
    print()

    # (B, N, 1), map labels to matched predictions
    print(src_logits.shape, src_idx)
    src_logits = src_logits[src_idx]
    src_logits = src_logits.squeeze(-1)
    target_classes[tgt_idx] = target_classes_o
    
    print(src_logits)
    print()


    src_masks = outputs["pred_masks"]
    src_masks = src_masks[src_idx]

    print(src_logits.cpu().detach().numpy())

    visualize_grid_v2(
        masks=src_masks.cpu().detach().numpy(),
        titles=src_logits.cpu().detach().numpy(),
        ncols=3,
        path="matcher_pred.jpg"
    )

    target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
    target_masks = target_masks.to(src_masks)
    target_masks = target_masks[tgt_idx]
    print(target_masks.shape)

    visualize_grid_v2(
        masks=target_masks.cpu().detach().numpy(),
        titles=src_logits.cpu().detach().numpy(),
        ncols=3,
        path="matcher_gt.jpg"
    )

    raise


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



