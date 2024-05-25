import torch 
from torch import nn

from models.seg.heads.instance_head import InstanceBranch
from models.seg.heads.mask_head import MaskBranch



class BaseDecoder(nn.Module):
    def __init__(self, in_channels, num_masks=10, output_iam=False):
        super().__init__()
        # add 2 for coordinates
        in_channels = in_channels

        self.scale_factor = 2
        self.output_iam = output_iam

        self.inst_branch = InstanceBranch(in_channels, num_masks)
        self.mask_branch = MaskBranch(in_channels)

    @torch.no_grad()
    def compute_coordinates_linspace(self, x):
        # linspace is not supported in ONNX
        h, w = x.size(2), x.size(3)
        y_loc = torch.linspace(-1, 1, h, device=x.device)
        x_loc = torch.linspace(-1, 1, w, device=x.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = -1.0 + 2.0 * torch.arange(h, device=x.device) / (h - 1)
        x_loc = -1.0 + 2.0 * torch.arange(w, device=x.device) / (w - 1)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    def forward(self, features):
        # coord_features = self.compute_coordinates(features)
        # features = torch.cat([coord_features, features], dim=1)
        
        pred_logits, pred_kernel, pred_scores, iam = self.inst_branch(features)
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

        # possible mask refinement
#         pred_masks = F.interpolate(
#             pred_masks, scale_factor=self.scale_factor,
#             mode='bilinear', align_corners=False)

#         output = {
#             "pred_logits": pred_logits,
#             "pred_masks": pred_masks,
#             "pred_scores": pred_scores,
#         }

        if self.output_iam:
            return pred_logits, pred_masks, pred_scores, iam

        return pred_logits, pred_masks, pred_scores