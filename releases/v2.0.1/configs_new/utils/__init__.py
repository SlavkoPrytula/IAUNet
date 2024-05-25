import yaml
from pathlib import Path
import logging.config

LOGGING_NAME = 'sparseinst'

TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format

# Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
DEFAULT_CONFIG = ROOT / "configs/default.yaml"


from os.path import join

class Image:
    size: int   = 1080

class Project:
    home_dir: str    = '/gpfs/space/home/prytula'
    work_dir: str    = join(home_dir, 'data/mask_labels/x63_fl') # home_dir + data_dir
    project_id: str = 'project-2-at-2022-05-20-09-23-83cd15f1'

class Config:
    image: Image           = Image

    # ---------------
    # coco masks
    coco_dataset: str      = join(Project.work_dir, f'coco/{Project.project_id}')

    # ---------------
    # original data
    bf_images: str         = join(Project.work_dir, f'np/plane_images/{Project.project_id}/bf_lower_higher')
    bf_image_name: str     = f'X_{Image.size}_bf_lower_higher_v1'
    masks: str             = join(Project.work_dir, f'np/multi_headed_segmentation/4_channel_segmentation/{Project.project_id}')
    masks_name: str        = f'Y_{Image.size}_bordered_masks_bsz-4_v1'

    
    # ---------------
    # fl
    fl_masks: str          = '/gpfs/space/projects/PerkinElmer/exhaustive_dataset/exhaustive_dataset_gt/acapella/63x_water_nc_41FOV_NIH3T3/Nuclei'
    dataset_x63_dir: str   = join('/gpfs/space/projects/PerkinElmer/exhaustive_dataset/PhaseImagesDL/', 'e696ed04-bec0-4061-adc4-4ee935973439/')
    csv_dataset_dir: str   = join(dataset_x63_dir, '63x_water_nc_41FOV.csv')



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
    yaml_save(file / 'default.yaml', cfg.__dict__)


def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    level = logging.INFO if verbose else logging.ERROR
    
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            name: {
                "format": "%(message)s"}},
        "handlers": {
            name: {
                "class": "logging.StreamHandler",
                "formatter": name,
                "level": level}},
        "loggers": {
            name: {
                "level": level,
                "handlers": [name],
                "propagate": False}}})


# Set logger
set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally


if __name__ == "__main__":
    cfg = get_config(DEFAULT_CONFIG)
    print(cfg.epochs)