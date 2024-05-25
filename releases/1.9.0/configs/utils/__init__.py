import yaml
from pathlib import Path
import logging.config

LOGGER = None
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format

# Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
DEFAULT_CONFIG = ROOT / "default.yaml"


class Config:
    def __init__(self, entries):
        self.__dict__.update(**entries)
        print(self.__dict__)

    def __setattr__(self, name, value):
        self.__dict__.update({name: value})

    def __repr__(self) -> str:
        return str(self.__dict__)


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
    cfg = Config(data)

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