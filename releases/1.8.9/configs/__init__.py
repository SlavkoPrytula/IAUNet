import inspect
import logging.config

from .base import cfg
from .datasets import Rectangle, Brightfield, Brightfield_Nuc
from .utils import set_logging

LOGGING_NAME = cfg.model.arch


# FIX: runtime dataset loading
datasets = {
    "brightfield": Brightfield, 
    "brightfield_nuc": Brightfield_Nuc, 
    "rectangle": Rectangle
}
cfg.dataset = datasets[cfg.dataset]



# print(cfg)