from omegaconf import OmegaConf
from .structure import cfg

import os
from os.path import join
import re


PROJECT_DIR = os.getcwd()
MODEL_FILES = join(PROJECT_DIR, "models/seg")
CONFIG_FILES = join(PROJECT_DIR, "configs")
UTILS_FILES = join(PROJECT_DIR, "utils")


def experiment_name(model_type):
    split_parts = re.split(r"[/-]", model_type)
    return "/".join(f"[{i}]" for i in split_parts)

OmegaConf.register_new_resolver("experiment_name", experiment_name)
