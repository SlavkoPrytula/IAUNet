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


class _dict(dict):
    """
    Custom dict wrapper to enable dot notation

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

# def my_dict_representer(dumper, data):
#     return dumper.represent_dict(data.items())

# yaml.add_representer(DICT, my_dict_representer)

class BaseConfig:
    # @staticmethod
    def get_class_attributes(self, class_obj):
        attributes = {}
        class_members = inspect.getmembers(class_obj)
        for member_name, member_value in class_members:
            if not member_name.startswith('__') and not inspect.ismethod(member_value):
                if inspect.isclass(member_value):
                    attributes[member_name] = self.get_class_attributes(member_value)
                else:
                    attributes[member_name] = member_value
        return attributes

    # @staticmethod
    # def create_cfg_dict(self):
    #     cfg_dict = {}
    #     cfg_members = inspect.getmembers(self)
    #     for member_name, member_value in cfg_members:
    #         if inspect.isclass(member_value):
    #             attributes = self.get_class_attributes(member_value)
    #             cfg_dict[member_name] = attributes
    #         # elif isinstance(member_value, dict):  # Check if the member is of type dict
    #         #     cfg_dict[member_name] = member_value
    #     return cfg_dict


    # def create_cfg_dict(self):
    #     cfg_dict = {}
    #     cfg_members = inspect.getmembers(self)
    #     for member_name, member_value in cfg_members:
    #         if inspect.isclass(member_value):
    #             attributes = self.get_class_attributes(member_value)
    #             cfg_dict[member_name] = attributes
    #         elif isinstance(member_value, dict):  # Check if the member is of type dict
    #             print('here', member_value)
    #             print(member_value.keys())
    #             print(member_value.values())

    #     return cfg_dict

    def create_cfg_dict(self):
        cfg_dict = {}
        cfg_members = inspect.getmembers(self)
        for member_name, member_value in cfg_members:
            if inspect.isclass(member_value):
                attributes = self.get_class_attributes(member_value)
                cfg_dict[member_name] = attributes
        
        # for k, v in cfg_dict
        # elif isinstance(member_value, dict):  # Check if the member is of type dict
        #     print('here', member_value)
        #     print(type(member_value))
        #     cfg_dict[member_name] = self._handle_dict(member_value)  # Use helper function to handle dict

        cfg_dict = {k: v for k, v in cfg_dict.items() if not k.startswith('__')}
        # print(cfg_dict)
        # print(inspect.isclass(cfg_dict["model"]["criterion"]))
        # print(isinstance(cfg_dict["model"]["criterion"], object))
        return cfg_dict

    # def _handle_dict(self, input_dict):
    #     result_dict = {}
    #     for key, value in input_dict.items():
    #         if isinstance(value, dict):  # Check if value is also a dict
    #             result_dict[key] = self._handle_dict(value)  # Recursively handle nested dicts
    #         else:
    #             result_dict[key] = value
    #     print(result_dict)
    #     print(type(result_dict))
    #     return result_dict

    # @staticmethod
    # def _handle_dict(input_dict):
    #     result_dict = {}
    #     for key, value in input_dict.items():
    #         if isinstance(value, dict):  # Check if value is also a dict
    #             result_dict[key] = BaseConfig._handle_dict(value)  # Recursively handle nested dicts
    #         else:
    #             result_dict[key] = value
    #     return result_dict


    def __dict__(self):
        cfg_dict = self.create_cfg_dict()
        return cfg_dict
    

    @staticmethod
    def _convert_path_to_str(obj):
        if isinstance(obj, Path):
            return str(obj)
        return obj
    
    def __repr__(self):
        return str(json.dumps(
            self.__dict__(), 
            default=self._convert_path_to_str, 
            sort_keys=True, 
            indent=4
            ))
    
    @classmethod
    def yaml_load(cls, file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        cls._update_attributes(cls, data)

    @staticmethod
    def _update_attributes(obj, data):
        for key, value in data.items():
            if hasattr(obj, key):
                if isinstance(value, dict):
                    nested_obj = getattr(obj, key)
                    obj._update_attributes(nested_obj, value)
                else:
                    setattr(obj, key, value)


# class dict(dict):
#     """
#     Custom dict wrapper to enable dot notation

#     Example:
#         >>> data = dict(attr_1="a", attr_2="b")
#         >>> data.attr_1
#         "a"
#         >>> data.attr_2
#         "b"
#     """
#     __getattr__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__



def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        dict: YAML data and file name.
    """
    with open(file, errors='ignore') as f:
        # Add YAML filename to dict and return
        return {**yaml.safe_load(f), 'yaml_file': str(file)} if append_filename else yaml.safe_load(f)


def yaml_save(file='data.yaml', data=None):
    """
    Save YAML data to a file.
    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict, optional): Data to save in YAML format. Default is None.
    Returns:
        None: Data is saved to the specified file.
    """
    file = Path(file)
    if not file.parent.exists():
        # Create parent directories if they don't exist
        file.parent.mkdir(parents=True, exist_ok=True)

    with open(file, 'w') as f:
        # Dump data to file in YAML format, converting Path objects to strings
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)


def get_config(file='data.yaml'):
    data = yaml_load(file)
    cfg = BaseConfig(data)

    return cfg


def save_config(cfg, file='data.yaml'):
    # yaml_save(ROOT / file, cfg)
    yaml_save(file / 'default.yaml', cfg.__dict__())


# def set_logging(name, verbose=True):
#     # sets up logging for the given name
#     level = logging.INFO if verbose else logging.ERROR
    
#     logging.config.dictConfig({
#         "version": 1,
#         "disable_existing_loggers": False,
#         "formatters": {
#             name: {
#                 "format": "%(message)s"}},
#         "handlers": {
#             name: {
#                 "class": "logging.StreamHandler",
#                 "formatter": name,
#                 "level": level}},
#         "loggers": {
#             name: {
#                 "level": level,
#                 "handlers": [name],
#                 "propagate": False}}})


def set_logging(name, log_file, verbose=True):
    level = logging.INFO if verbose else logging.ERROR
    
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            name: {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}},
        "handlers": {
            name: {
                "class": "logging.FileHandler", 
                "filename": log_file, 
                "formatter": name,
                "level": level}},
        "loggers": {
            name: {
                "level": level,
                "handlers": [name],
                "propagate": False}}})
    

    LOGGER = logging.getLogger(name)  # define globally



if __name__ == "__main__":
    cfg = get_config(DEFAULT_CONFIG)
    print(cfg.epochs)