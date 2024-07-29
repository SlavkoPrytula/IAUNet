import sys
sys.path.append("./")

from utils.augmentations import normalize
from utils.augmentations import train_transforms
from utils.registry import DATASETS_CFG, DATASETS
from configs import cfg
from dataset.dataloaders import build_loader_ms, trivial_batch_collator

cfg.dataset = DATASETS_CFG.get("LiveCell")
dataset = DATASETS.get(cfg.dataset.type)

train_dataset = dataset(cfg, 
                        dataset_type="train", 
                        normalization=normalize, 
                        transform=train_transforms(cfg)
                        )

print(len(train_dataset))

train_dataloader = build_loader_ms(train_dataset, 
                                batch_size=cfg.train.batch_size, 
                                num_workers=0, 
                                collate_fn=trivial_batch_collator)


loader = iter(train_dataloader)

batch = next(loader)
for i in range(len(batch)):
    target = batch[i]
    print(target["image"].shape)

print()
batch = next(loader)
for i in range(len(batch)):
    target = batch[i]
    print(target["image"].shape)

print()
batch = next(loader)
for i in range(len(batch)):
    target = batch[i]
    print(target["image"].shape)