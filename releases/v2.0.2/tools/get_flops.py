from thop import profile
import torch

import sys
sys.path.append("./")

from configs import cfg
from utils.registry import MODELS
from models.build_model import build_model, get_model


cfg.model.kernel_dim = 256
cfg.model.n_levels = 5

model = get_model(cfg)
input_tensor = torch.randn(1, 1, 512, 512)

flops, params = profile(model, inputs=(input_tensor, ), verbose=False)

gflops = flops / 1e9
params_in_millions = params / 1e6

print(f'Flops: {gflops:.3f}G')
print(f'Params: {params_in_millions:.3f}M')
