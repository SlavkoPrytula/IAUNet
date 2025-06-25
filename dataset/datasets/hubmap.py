import sys
sys.path.append('.')

from torch.utils.data import Dataset
from configs import cfg

from utils.registry import DATASETS


@DATASETS.register(name="HuBMAP")
class HuBMAP(Dataset):
    def __init__(self, cfg: cfg, dataset_type="train", transform=None, **kwargs):
        super().__init__(cfg, dataset_type=dataset_type, transform=transform, **kwargs)
