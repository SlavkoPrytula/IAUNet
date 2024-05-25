import torch
import torch.nn.functional as F


# def unfold(input, kernel_size=256, stride=256):
#     c = input.shape[0]
#     patches = input.unfold(1, kernel_size, stride).unfold(2, kernel_size, stride)
#     patches = patches.contiguous().view(c, -1, kernel_size, kernel_size)
#     patches = patches.permute(1, 0, 2, 3)

#     return patches


def unfold(input, kernel_size=256, stride=256):
    B, C, H, W = input.shape
    patches = input.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    patches = patches.contiguous().view(B, C, -1, kernel_size, kernel_size)
    patches = patches.permute(2, 0, 1, 3, 4)

    return patches


def fold(patches, out_size, b=1, kernel_size=256, stride=256):
    c = patches.shape[1]

    patches = patches.contiguous().transpose(1, 0).view(b, c, -1, kernel_size*kernel_size)      # [B, C, n_patches, kernel_size*kernel_size]
    patches = patches.permute(1, 0, 3, 2)                                                       # [B, C, kernel_size*kernel_size, n_patches]
    patches = patches.contiguous().view(b, c*kernel_size*kernel_size, -1)                       # (B, C*kernel*kernel, n_patches)

    recovery_mask = F.fold(torch.ones_like(patches),
                           output_size=out_size,
                           kernel_size=kernel_size, stride=stride)
    out = F.fold(patches, out_size,
                   kernel_size=kernel_size, stride=stride)                               # [B, C, H, W]
    out /= recovery_mask

    return out