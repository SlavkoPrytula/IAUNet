from os.path import join
from .base import Project, Image


class Dataset:
    name: str
    coco_dataset: str


class Rectangle(Dataset):
    name: str = 'rectangle'
    coco_dataset: str      = join(Project.home_dir, f'data/datasets/synthetic_datasets/rectangle/rectangles_[n=100]_[R_min=5_R_max=15]_[30.06.23].json')