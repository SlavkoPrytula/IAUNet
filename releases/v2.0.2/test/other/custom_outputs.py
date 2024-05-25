import torch 
from torch import nn
from torch.nn import functional as F
import numpy as np



kernel = {
    "a": 1,
    "b": 1,
}

iam = {
    "c": 2,
    "d": 2
}

for k, i in zip(kernel[:-1], iam[:-1]):
    print(k, i)

# for i in iam["c"]:
#     print(i)

# l = list(iam.keys())[-1]
# kernel[l] = iam[l]
# print()
