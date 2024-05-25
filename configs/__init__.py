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
    # from utils.registry import DATASETS_CFG
    cfg.dataset = DATASETS_CFG.get(cfg.dataset.name)
except ImportError:
    print("WARNING: Could not import 'from configs.datasets import DATASETS_CFG' properly!")

    # try:
    #     print("WARNING: Trying sys.import")
    #     import sys
    #     sys.path.append(".")
    #     from configs.datasets import DATASETS_CFG
    #     cfg.dataset = DATASETS_CFG.get(cfg.dataset.name)
    # except ImportError:
    #     print("WARNING: Could not import 'from configs.datasets import DATASETS_CFG' properly with 'sys.import'!")
    #     print("WARNING: Import failed!")


from utils.logging import setup_logger
logger = setup_logger(name=LOGGING_NAME)