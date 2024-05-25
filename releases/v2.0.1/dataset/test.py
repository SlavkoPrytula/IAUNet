import sys
sys.path.append("./")

from catalog import *
from datasets.rectangle import Rectangle_Dataset
from datasets.brightfiled import Brightfield_Dataset
from dataloaders import build_loader

from configs import cfg


DatasetCatalog.register("rectangle", Rectangle_Dataset)
DatasetCatalog.register("brightfiled", Brightfield_Dataset)

dataset = DatasetCatalog.get("brightfiled")
dataset = dataset(cfg, "train")

rectangle_dataset = build_loader(
    dataset, batch_size=1, num_workers=2
)

print(DatasetCatalog)