import torch 
from torch import nn
from torch.nn import functional as F
import numpy as np

import sys
sys.path.append("./")

from utils.tiles import fold, unfold



# mask = torch.rand(2, 256, 512, 512)
# mask1 = mask.clone()
# mask = unfold(mask, kernel_size=64, stride=32)
# N, B, C, H, W = mask.shape
# mask = mask.contiguous()
# print(mask.shape)

# mask = mask.view(N, B*C, H, W)
# print(mask.shape)

# mask = fold(mask, out_size=512, kernel_size=64, stride=32)
# print(mask.shape)

# _, _, H, W = mask.shape

# mask = mask.view(B, C, H, W)
# print(mask.shape)

# print(torch.all(mask == mask1))


# a = torch.rand(10, 1, 512, 512).flatten(1)
# print(a.shape)