import yaml
from pathlib import Path
import logging.config

import inspect
import json 
import yaml

LOGGER = None
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format

# Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
DEFAULT_CONFIG = ROOT / "default.yaml"


class dict(dict):
    """
    Custom dict wrapper to enable dot notation and type hinting.

    Example:
        >>> data = dict(attr_1="a", attr_2="b")
        >>> data.attr_1
        "a"
        >>> data.attr_2
        "b"
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



def update_dependent_values(instance: dict, old_values: dict, new_values: dict):
    for key, value in instance.items():
        if isinstance(value, dict):
            update_dependent_values(value, old_values, new_values)
        elif isinstance(value, str) and value in old_values.values():
            for old_key, old_val in old_values.items():
                if value == old_val and old_key in new_values:
                    instance[key] = new_values[old_key]

def find_matching_subdict(instance: dict, template: dict) -> dict:
    for key, value in instance.items():
        if isinstance(value, dict) and set(value.keys()) == set(template.keys()):
            return value
        elif isinstance(value, dict):
            result = find_matching_subdict(value, template)
            if result:
                return result
    return None

def merge_cfg(instance: dict, new_class: dict):
    old_class = find_matching_subdict(instance, new_class)
    if old_class:
        for key, value in instance.items():
            if isinstance(value, dict) and set(value.keys()) == set(new_class.keys()):
                setattr(instance, key, new_class)
        update_dependent_values(instance, old_class, new_class)


