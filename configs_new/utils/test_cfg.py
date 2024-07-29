import sys
sys.path.append("./")
from configs.base.default_runtime import cfg
from configs.models.iaunet_resnet.iaunet_r50 import backbone


# merge_cfg(model, backbone)
# print(model)

print(cfg.model)