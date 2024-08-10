import sys
sys.path.append('.')

import hydra
from configs import cfg

from dataset.datasets.base_coco_dataset import BaseCOCODataset
from utils.registry import DATASETS


@DATASETS.register(name="LiveCell")
class LiveCell(BaseCOCODataset):
    def __init__(self, cfg: cfg, dataset_type="train", normalization=None, transform=None):
        super().__init__(cfg, dataset_type, normalization, transform)


@DATASETS.register(name="LiveCellCrop")
class LiveCellCrop(LiveCell):
    def __init__(self, cfg: cfg, dataset_type="train", normalization=None, transform=None):
        super().__init__(cfg, dataset_type, normalization, transform)


@DATASETS.register(name="LiveCell2Percent")
class LiveCell2Percent(LiveCell):
    def __init__(self, cfg: cfg, dataset_type="train", normalization=None, transform=None):
        super().__init__(cfg, dataset_type, normalization, transform)
        self.total_size = 100 if dataset_type == "valid" else self.total_size


@DATASETS.register(name="LiveCell30Images")
class LiveCell30Images(LiveCell):
    def __init__(self, cfg: cfg, dataset_type="train", normalization=None, transform=None):
        super().__init__(cfg, dataset_type, normalization, transform)
        self.total_size = 50 if dataset_type == "train" else 10

        

@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: cfg):
    from utils.visualise import visualize
    from utils.utils import flatten_mask
    from utils.augmentations import normalize
    from utils.augmentations import train_transforms, valid_transforms
    import time

    time_s = time.time()

    dataset = LiveCell(cfg, dataset_type="train", 
                    #   normalization=normalize,
                      transform=train_transforms(cfg)
                      )
    
    print(len(dataset))
    
    targets = dataset[5]
    time_e = time.time()
    print(f'loaded in {time_e - time_s}(s)')

    print(targets["image"].shape, targets["ori_shape"])
    print(targets["labels"])

    print(targets["image"].shape, targets["instance_masks"].shape)
    print(f'std: {targets["image"].std()}, mean: {targets["image"].mean()}')

    visualize(
        images=targets["image"][0, ...], 
        path='./test_image.jpg', 
        cmap='gray'
    )
    visualize(
        images=flatten_mask(targets["instance_masks"].numpy(), axis=0)[0], 
        path='./test_mask.jpg', 
        cmap='gray'
    )

if __name__ == "__main__":
    main()
