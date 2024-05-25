# import inspect
# import logging.config

# from .base import cfg, TIME
# from .datasets import (
#     Rectangle, 
#     Brightfield, 
#     Brightfield_Nuc, 
#     Synthetic_Brightfield, 
#     OriginalPlusSynthetic_Brightfield
#     )
# from .utils import set_logging, LOGGER

# LOGGING_NAME = cfg.model.arch
# MODEL_FILES = "/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/models/seg"

# # FIX: runtime dataset loading
# datasets = {
#     "brightfield": Brightfield, 
#     "brightfield_nuc": Brightfield_Nuc, 
#     "synthetic_brightfield": Synthetic_Brightfield, 
#     "original_plus_synthetic_brightfield": OriginalPlusSynthetic_Brightfield,
#     "rectangle": Rectangle
# }
# cfg.dataset = datasets[cfg.dataset]

import os
from os.path import join
from .utils import set_logging, LOGGER
from .base import cfg

LOGGING_NAME = cfg.model.arch
MODEL_FILES = join(os.getcwd(), "models/seg")


from .datasets import DATASETS_CFG
cfg.dataset = DATASETS_CFG.get(cfg.dataset)