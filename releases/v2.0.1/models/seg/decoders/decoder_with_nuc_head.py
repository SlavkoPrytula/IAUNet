import torch 
from torch import nn

from models.seg.heads.nuceli_head import NuceliHead
from models.seg.heads.mask_head import MaskHead



class DecoderWithNuc(nn.Module):
    def __init__(self, in_channels, num_masks=10):
        super().__init__()
        in_channels = in_channels

        self.nuc_branch = NuceliHead(in_channels, num_masks)
        self.mask_branch = MaskHead(in_channels)


    def forward(self, features):
        pred_logits, pred_kernel, pred_scores, pred_nuc = self.nuc_branch(features)
        mask_features = self.mask_branch(features)
        
        # Predicting instance masks
        N = pred_kernel.shape[1]  # num_masks
        
        B, C, H, W = mask_features.shape
        mask_features = mask_features.view(B, C, -1)   # (B, C, H, W) -> (B, C, [HW])
        pred_masks = torch.matmul(
            pred_kernel,    # (B, N, 128)
            mask_features   # (B, 128, [HW])
        ) # -> (B, N, [HW])
        pred_masks = pred_masks.view(B, N, H, W)  # (B, N, [HW]) -> (B, N, H, W)

        return pred_logits, pred_masks, pred_scores, pred_nuc