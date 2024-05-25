import inspect
import logging.config

from .base import cfg, TIME
from .datasets import (
    Rectangle, 
    Brightfield, 
    Brightfield_Nuc, 
    Synthetic_Brightfield
    )
from .utils import set_logging, LOGGER

LOGGING_NAME = cfg.model.arch
MODEL_FILES = "<path to project>/models/seg"

# FIX: runtime dataset loading
datasets = {
    "brightfield": Brightfield, 
    "brightfield_nuc": Brightfield_Nuc, 
    "synthetic_brightfield": Synthetic_Brightfield, 
    "rectangle": Rectangle
}
cfg.dataset = datasets[cfg.dataset]



# print(cfg)