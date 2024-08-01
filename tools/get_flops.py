from thop import profile
import torch

import sys
sys.path.append("./")

from configs import cfg
from utils.registry import MODELS
from models.build_model import get_model

# from calflops import calculate_flops
# https://github.com/MrYxJ/calculate-flops.pytorch


cfg.model.instance_head.type = "InstanceHead-v1.1"
# cfg.model.instance_head.type = "InstanceHead-v1.2-occluders"
# cfg.model.instance_head.type = "InstanceHead-v1.3-overlaps"
# cfg.model.instance_head.type = "InstanceHead-v3-multiheaded"
# cfg.model.instance_head.type = "InstanceHead-v2.0-overlaps-attn"
# cfg.model.instance_head.type = "Refiner"
cfg.model.instance_head.in_channels = 256
cfg.model.instance_head.kernel_dim = 256
# cfg.model.instance_head.num_convs = 2
cfg.model.instance_head.num_groups = 1
cfg.model.instance_head.num_masks = 100
# cfg.model.instance_head.activation = "sigmoid"
cfg.model.mask_dim = 256
# cfg.model.inst_dim = 256
cfg.model.num_convs = 4
cfg.model.n_levels = 4

cfg.model.type = "iaunet"
# cfg.model.type = "iaunet_occluders"
# cfg.model.type = "iaunet_ml"
# cfg.model.type = "custom/truncated_decoder/iaunet"

cfg.model.backbone.out_indices = [1, 2, 3, 4]


model = get_model(cfg)
input_tensor = torch.randn(1, 3, 512, 512)

if torch.cuda.is_available():
    model = model.to("cuda:0")
    input_tensor = input_tensor.to("cuda:0")
model.eval()


# get-flops.
flops, params = profile(model, inputs=(input_tensor,), verbose=True)

gflops = flops / 1e9
params_in_millions = params / 1e6
print(f'Flops: {gflops:.3f}G')
print(f'Params: {params_in_millions:.3f}M')
