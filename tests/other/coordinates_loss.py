import torch 
from torch import nn
from torch.nn import functional as F
import numpy as np


def center_of_mass(bitmasks):
    _, h, w = bitmasks.size()
    ys = torch.arange(0, h, dtype=torch.float32, device=bitmasks.device)
    xs = torch.arange(0, w, dtype=torch.float32, device=bitmasks.device)

    m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
    m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
    m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
    center_x = m10 / m00
    center_y = m01 / m00
    return center_x, center_y




a = np.zeros((2, 10, 10))
a[0, :5, :5] = 1
a[1, 7:, 0:] = 1
print(a.shape)

a = torch.tensor(a)


x, y = center_of_mass(a)
b = torch.stack([x, y]).t()

print(x, y)

print(b.shape)



b = np.zeros((2, 10, 2))
for i in b[0, :]:
    print(f"{i}")


print(np.round(torch.rand(3, 2).numpy(), 1))

# (N, 2)