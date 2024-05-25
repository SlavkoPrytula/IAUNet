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


@MATCHERS.register(name="HungarianMatcher")
class HungarianMatcher(nn.Module):
    def __init__(self, cfg: cfg):
        super().__init__()
        # self.alpha = cfg.model.criterion.matcher.mask_cost
        # self.beta = cfg.model.criterion.matcher.cls_cost
        
        self.mask_cost = cfg.mask_cost
        self.cls_cost = cfg.cls_cost
        
        self.mask_score = dice_score

    def forward(self, outputs, targets, input_shape):
        with torch.no_grad():
            B, N, H, W = outputs["pred_masks"].shape

            pred_masks = outputs['pred_masks']
            pred_logits = outputs['pred_logits'].flatten(0, 1).sigmoid()
            # pred_logits = outputs['pred_logits'].sigmoid()
            # print(pred_logits.shape)
            
            tgt_ids = torch.cat([v["labels"] for v in targets])

            if tgt_ids.shape[0] == 0:
                return [(torch.as_tensor([]).to(pred_logits), torch.as_tensor([]).to(pred_logits))] * B
            
            masks = [t["masks"] for t in targets]
            # target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
            # target_masks = target_masks.to(pred_masks)
            
            target_masks, valid = nested_masks_from_list(masks, input_shape).decompose()
            target_masks = target_masks.to(pred_masks)

            # tgt_masks = F.interpolate(
            #     tgt_masks[:, None], size=pred_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

            pred_masks = pred_masks.view(B * N, -1)
            target_masks = target_masks.flatten(1)
            # print(pred_masks.shape, target_masks.shape)
            # pred_masks = pred_masks.flatten(1)
            # target_masks = target_masks.flatten(1)


            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (
                (1 - alpha) * (pred_logits ** gamma) * (-(1 - pred_logits + 1e-8).log())
            )
            pos_cost_class = (
                alpha * ((1 - pred_logits) ** gamma) * (-(pred_logits + 1e-8).log())
            )
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]


            with amp.autocast(enabled=False):
                pred_masks = pred_masks.float()
                target_masks = target_masks.float()
                pred_logits = pred_logits.float()

                mask_score = -self.mask_score(pred_masks, target_masks)
                # print(mask_score.shape)
                # print(cost_class.shape)
                # print(outputs['pred_logits'][:, tgt_ids].shape)
                
                # Nx(Number of gts)
                # cost_class = pred_logits.view(B * N, -1)[:, tgt_ids]
                # C = (mask_score ** self.mask_cost) * (matching_prob ** self.cls_cost)
                C = (mask_score * self.mask_cost) + (cost_class * self.cls_cost)

            C = C.view(B, N, -1).cpu()
            C = torch.nan_to_num(C, nan=0, posinf=0, neginf=0) # FIXME:

            # hungarian matching
            sizes = [len(v["masks"]) for v in targets]
            # indices = [linear_sum_assignment(c[i], maximize=True)
            #            for i, c in enumerate(C.split(sizes, -1))]
            # indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
            # return indices

            indices = [
                linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
            ]
            return [
                (
                    torch.as_tensor(i, dtype=torch.int64),
                    torch.as_tensor(j, dtype=torch.int64),
                )
                for i, j in indices
            ]
        



# class HungarianMatcherIAM(nn.Module):
    # def __init__(self, cfg: cfg):
    #     super().__init__()
    #     self.alpha = cfg.model.matcher.alpha
    #     self.beta = cfg.model.matcher.beta
    #     self.mask_score = dice_score

    # def forward(self, outputs, targets, input_shape):
    #     with torch.no_grad():
    #         B, N, H, W = outputs["pred_masks"].shape

    #         pred_masks = outputs['pred_masks']
    #         pred_logits = outputs['pred_logits'].sigmoid()
    #         tgt_ids = torch.cat([v["labels"] for v in targets])

    #         if tgt_ids.shape[0] == 0:
    #             return [(torch.as_tensor([]).to(pred_logits), torch.as_tensor([]).to(pred_logits))] * B
            
    #         tgt_masks, _ = nested_masks_from_list([t["masks"] for t in targets], input_shape).decompose()
    #         tgt_masks = tgt_masks.to(pred_masks)

    #         tgt_masks = F.interpolate(tgt_masks[:, None], size=pred_masks.shape[-2:], 
    #                                   mode="bilinear", align_corners=False).squeeze(1)

    #         pred_masks = pred_masks.view(B * N, -1)
    #         tgt_masks = tgt_masks.flatten(1)

    #         with amp.autocast(enabled=False):
    #             pred_masks = pred_masks.float()
    #             tgt_masks = tgt_masks.float()
    #             pred_logits = pred_logits.float()

    #             mask_score = self.mask_score(pred_masks, tgt_masks)

    #             # Nx(Number of gts)
    #             matching_prob = pred_logits.view(B * N, -1)[:, tgt_ids]

    #             print(mask_score.shape)
    #             print(matching_prob.shape)
                
    #             C = (mask_score ** self.mask_cost) * (matching_prob ** self.cls_cost)

    #         C = C.view(B, N, -1).cpu()

    #         # hungarian matching
    #         sizes = [len(v["masks"]) for v in targets]
    #         indices = [linear_sum_assignment(c[i], maximize=True)
    #                    for i, c in enumerate(C.split(sizes, -1))]
            
    #         indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    #         return indices



@MATCHERS.register(name="IAM_Matcher")
class HungarianMatcherIAM(nn.Module):
    def __init__(self, cfg: cfg):
        super().__init__()
        self.alpha = cfg.model.matcher.alpha
        self.beta = cfg.model.matcher.beta
        self.mask_score = dice_score

    def forward(self, outputs, targets, input_shape):
        with torch.no_grad():
            B, N, H, W = outputs["pred_iam"].shape

            pred_masks = outputs['pred_iam']
            pred_logits = outputs['pred_logits'].sigmoid()
            tgt_ids = torch.cat([v["iam_labels"] for v in targets])

            if tgt_ids.shape[0] == 0:
                return [(torch.as_tensor([]).to(pred_logits), torch.as_tensor([]).to(pred_logits))] * B
            
            tgt_masks, _ = nested_masks_from_list([t["iam_masks"] for t in targets], input_shape).decompose()
            tgt_masks = tgt_masks.to(pred_masks)

            tgt_masks = F.interpolate(tgt_masks[:, None], size=pred_masks.shape[-2:], 
                                      mode="bilinear", align_corners=False).squeeze(1)

            pred_masks = pred_masks.view(B * N, -1)
            tgt_masks = tgt_masks.flatten(1)

            with amp.autocast(enabled=False):
                pred_masks = pred_masks.float()
                tgt_masks = tgt_masks.float()
                pred_logits = pred_logits.float()

                mask_score = self.mask_score(pred_masks, tgt_masks)
                
                # Nx(Number of gts)
                matching_prob = pred_logits.view(B * N, -1)[:, tgt_ids]
                C = (mask_score ** self.alpha) * (matching_prob ** self.beta)

            C = C.view(B, N, -1).cpu()

            # hungarian matching
            sizes = [len(v["iam_masks"]) for v in targets]
            indices = [linear_sum_assignment(c[i], maximize=True)
                       for i, c in enumerate(C.split(sizes, -1))]
            
            indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
            return indices




# class HungarianMatcher(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, preds, targets, pred_logits=None):
#         with torch.no_grad():
#             B, N, H, W = targets.shape
#             _targets = targets.view(B * N, -1)   # (B, N, H, W) -> ([BN], [HW])
#             B, N, H, W = preds.shape
#             _preds = preds.view(B * N, -1)       # (B, N, H, W) -> ([BN], [HW])
            
            
#             with amp.autocast(enabled=False):
#                 _preds = _preds.float()
#                 _targets = _targets.float()
#                 mask_score = dice_score(_preds, _targets)

#                 if pred_logits is not None:
#                     # NEW: testing
#                     matching_prob = pred_logits.view(B * N, -1)
#                     C = (mask_score ** 0.8) * (matching_prob ** 0.2)
#                 else:
#                     C = mask_score

#                 C = C.view(B, N, -1).cpu()
# #                 C = torch.nan_to_num(C, nan=0, posinf=0, neginf=0)

#             # hungarian matching
#             sizes = [len(a) for a in targets]
#             indices = [linear_sum_assignment(c[i], maximize=True) for i, c in enumerate(C.split(sizes, -1))]
#             indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
            
#             return indices


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
        "pred_logits": torch.tensor([[0.2, 0.3, 0.2, 0.1, 0.1], [0.5, 0.5, 0.6, 0.1, 0.1]]).unsqueeze(-1)
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
    # indices = matcher(outputs, targets, [10, 10])
    indices = matcher(outputs, targets)
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
    # print(list(zip(targets, indices)))
    print()
    # raise

    target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
    # target_labels = torch.cat([t["labels"] for t in targets])
    # target_labels = target_labels[tgt_idx[-1]]

    # print(target_labels)
    # print()
    
    # (B, N, 1)
    target_classes = torch.full(src_logits.shape[:2], num_classes,
                                dtype=torch.int64, device=src_logits.device)
    
    print(target_classes)
    print()

    # (B, N, 1), map labels to matched predictions
    print(src_logits.shape, src_idx)
    src_logits = src_logits[src_idx]
    target_classes[tgt_idx] = target_classes_o
    
    print(src_logits)
    print()


    src_masks = outputs["pred_masks"]
    src_masks = src_masks[src_idx]

    visualize_grid_v2(
        masks=src_masks.cpu().detach().numpy(),
        ncols=3,
        path="matcher_pred.jpg"
    )

    target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
    target_masks = target_masks.to(src_masks)
    target_masks = target_masks[tgt_idx]
    print(target_masks.shape)

    visualize_grid_v2(
        masks=target_masks.cpu().detach().numpy(),
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



