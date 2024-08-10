import os
from os.path import join
from configs.base import cfg


LOGGING_NAME = cfg.model.type
PROJECT_DIR = os.getcwd()
MODEL_FILES = join(PROJECT_DIR, "models/seg")
CONFIG_FILES = join(PROJECT_DIR, "configs")
UTILS_FILES = join(PROJECT_DIR, "utils")


try:
    from configs.datasets import DATASETS_CFG
    cfg.dataset = DATASETS_CFG.get(cfg.dataset.name)
except ImportError:
    print("WARNING: Could not import 'from configs.datasets import DATASETS_CFG' properly!")
