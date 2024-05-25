import os

# print(__name__.split('.'))
# print(os.path.dirname(__file__))

import importlib.util
import sys
from pathlib import Path

sys.path.append('.')

from configs import cfg
from models.build_model import build_model
from models import import_from_file


files = [
    Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=46036037]-[2023-08-09 13:25:51]"),
    Path("/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_add_overlaps]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'overlaps']]/[job=46034550]-[2023-08-09 13:00:46]"),
]

experiment = files[0]

# module_name = '-'
file_path = experiment / 'model_files/__init__.py'

# spec = importlib.util.spec_from_file_location(module_name, file_path)
# module = importlib.util.module_from_spec(spec)
# sys.modules[module_name] = module
# print(sys.modules[module_name])
# spec.loader.exec_module(module)

module = import_from_file(file_path, clear_cache=True)

model_class = getattr(module, 'SparseSEUnet')

config_path = experiment / "default.yaml"
cfg.yaml_load(config_path)

model = model_class(cfg)
try:
    print(model.overlap_conv)
except:
    print('couldnt load')


# del sys.modules[module_name]


importlib.invalidate_caches()
importlib.util.cache_from_source
# print(importlib.util)
experiment = files[1]

# module_name = 'SparseSEUnet'
file_path = experiment / 'model_files/__init__.py'

# spec = importlib.util.spec_from_file_location(module_name, file_path)
# module = importlib.util.module_from_spec(spec)
# sys.modules[module_name] = module
# print(sys.modules[module_name])
# spec.loader.exec_module(module)

module = import_from_file(file_path, clear_cache=True)

model_class = getattr(module, 'SparseSEUnet')

# importlib.reload(module)

config_path = experiment / "default.yaml"
cfg.yaml_load(config_path)

model = model_class(cfg)
try:
    print(model.overlap_conv)
except:
    print('couldnt load')

