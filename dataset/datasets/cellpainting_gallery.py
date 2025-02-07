import sys
sys.path.append('.')

import numpy as np
import hydra
from configs import cfg

from dataset.datasets.base_coco_dataset import BaseCOCODataset
from utils.registry import DATASETS


@DATASETS.register(name="cellpainting_gallery")
class CellPaintingGallery(BaseCOCODataset):
    def __init__(self, cfg: cfg, dataset_type="train", normalization=None, transform=None):
        super().__init__(cfg, dataset_type, normalization, transform)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: cfg):
    from utils.visualise import visualize, visualize_grid_v2
    from visualizations import visualize_masks
    from utils.augmentations import train_transforms, valid_transforms
    import time

    time_s = time.time()

    dataset = CellPaintingGallery(cfg, dataset_type="train", 
                                  transform=train_transforms(cfg))
    # for data in dataset:
    #     print(data['image'].shape, data['instance_masks'].shape)
    # raise

    print(len(dataset))
    
    targets = dataset[4]
    time_e = time.time()
    print(f'loaded in {time_e - time_s}(s)')

    print(targets["image"].shape, targets["ori_shape"])
    print(targets["labels"])
    print(len(targets["labels"]), len(targets["instance_masks"]))

    print(targets["image"].shape, targets["instance_masks"].shape)
    print(f'std: {targets["image"].std()}, mean: {targets["image"].mean()}')

    visualize(
        images=targets["image"][0, ...], 
        path='./test_image.jpg', 
        cmap='gray',
        show_title=False
    )

    # H, W = targets["image"].shape[-2:] #targets["ori_shape"]
    H, W = targets["ori_shape"]
    visualize_masks(
        figsize=[30, 30],
        img=targets["image"][0, ...],
        masks=targets["instance_masks"],
        shape=[H, W],
        alpha=0.65,
        draw_border=True, 
        static_color=False,
        path='./test_mask.jpg',
        dpi=100
        # show_img=True
    )

    # visualize_grid_v2(
    #     masks=targets["instance_masks"].numpy(), 
    #     path='./test_inst.jpg',
    #     ncols=5
    # )

if __name__ == "__main__":
    main()
