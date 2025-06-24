import torch
from torch import nn, Tensor
from torch.nn import functional as F


# mixup scaling [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7924688/]
class MixUpScaler(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.mixing = nn.Parameter(torch.tensor(0.5))
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.mixing * F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False) \
            + (1 - self.mixing) * F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic')
        return x
