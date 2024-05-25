import torch
import torch.nn.functional as F


def unfold(input, size=256, step=256):
    c = input.shape[0]
    patches = input.unfold(1, size, step).unfold(2, size, step)
    patches = patches.contiguous().view(c, -1, size, size)
    patches = patches.permute(1, 0, 2, 3)

    return patches


def fold(patches, out_size, b=1, size=256, step=256):
    c = patches.shape[1]

    patches = patches.contiguous().transpose(1, 0).view(b, c, -1, size*size)    # [B, C, n_patches, kernel_size*kernel_size]
    patches = patches.permute(1, 0, 3, 2)                                       # [B, C, kernel_size*kernel_size, n_patches]
    patches = patches.contiguous().view(b, c*size*size, -1)                     # (B, C*kernel*kernel, n_patches)

    recovery_mask = F.fold(torch.ones_like(patches),
                           output_size=out_size,
                           kernel_size=size, stride=step)
    out = F.fold(patches, out_size,
                   kernel_size=size, stride=step)                               # [B, C, H, W]
    out /= recovery_mask

    return out