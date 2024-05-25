import torch 
from torch import nn
import torch.nn.functional as F
from torch.cuda import amp

from scipy.optimize import linear_sum_assignment
from utils.utils import nested_masks_from_list
from utils.losses import dice_score


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
    batch_idx = torch.cat([torch.full_like(src, i)
                            for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def _get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i)
                            for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx
        


class HungarianMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.8
        self.beta = 0.2
#         self.alpha = 1
#         self.beta = 0 
        self.mask_score = dice_score

    def forward(self, outputs, targets, input_shape):
        with torch.no_grad():
            B, N, H, W = outputs["pred_masks"].shape

            pred_masks = outputs['pred_masks']
            pred_logits = outputs['pred_logits'].sigmoid()
            tgt_ids = torch.cat([v["labels"] for v in targets])

            if tgt_ids.shape[0] == 0:
                return [(torch.as_tensor([]).to(pred_logits), torch.as_tensor([]).to(pred_logits))] * B
            
            tgt_masks, _ = nested_masks_from_list([t["masks"] for t in targets], input_shape).decompose()

            device = pred_masks.device
            tgt_masks = tgt_masks.to(pred_masks)

            tgt_masks = F.interpolate(
                tgt_masks[:, None], size=pred_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

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
            sizes = [len(v["masks"]) for v in targets]
            indices = [linear_sum_assignment(c[i], maximize=True)
                       for i, c in enumerate(C.split(sizes, -1))]
            
            indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
            return indices




class HungarianMatcherIAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.8
        self.beta = 0.2
#         self.alpha = 1
#         self.beta = 0 
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

            device = pred_masks.device
            tgt_masks = tgt_masks.to(pred_masks)

            tgt_masks = F.interpolate(
                tgt_masks[:, None], size=pred_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

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

