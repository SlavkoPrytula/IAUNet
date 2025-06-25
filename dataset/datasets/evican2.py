import sys
sys.path.append('.')

import hydra
from configs import cfg
from dataset.datasets.base_coco_dataset import BaseCOCODataset
from utils.registry import DATASETS


@DATASETS.register(name="EVICAN2")
class EVICAN2(BaseCOCODataset):
    def __init__(self, cfg: cfg, dataset_type="train", normalization=None, transform=None):
        super().__init__(cfg, dataset_type, normalization, transform)
        self.filter_bg()
        self.total_size = len(self.image_ids)

    def filter_bg(self):
        self.image_ids = [
            img_id for img_id in self.image_ids 
            if "Background" not in self.coco.loadImgs([img_id])[0]['file_name']
        ]


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: cfg):
    from visualizations.visualise import visualize, visualize_grid_v2
    from visualizations import visualize_masks
    from utils.utils import flatten_mask
    from utils.augmentations import normalize
    from utils.augmentations import train_transforms, valid_transforms
    import time
    import numpy as np
    from tqdm import tqdm

    time_s = time.time()

    dataset = EVICAN2(cfg, dataset_type="eval", 
                    #   normalization=normalize,
                      transform=valid_transforms(cfg)
                      )
    
    print(len(dataset))

    targets = dataset[10]
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

    visualize(
        images=targets["image"][0, ...], 
        path='./test_image.jpg', 
        cmap='gray',
        show_title=False
    )

    H, W = targets["image"].shape[-2:] #targets["ori_shape"]
    # H, W = targets["ori_shape"]
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
