import sys
sys.path.append('.')

import hydra
from configs import cfg

from dataset.datasets.base_coco_dataset import BaseCOCODataset
from utils.registry import DATASETS


@DATASETS.register(name="Revvity_25")
class Revvity_25(BaseCOCODataset):
    def __init__(self, cfg: cfg, dataset_type="train", transform=None, **kwargs):
        super().__init__(cfg, dataset_type=dataset_type, transform=transform, **kwargs)
        

@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: cfg):
    from visualizations.visualise import visualize, visualize_grid_v2
    from visualizations import visualize_masks
    from utils.augmentations import train_transforms, valid_transforms
    import time

    time_s = time.time()

    dataset = Revvity_25(cfg, dataset_type="train", 
                         transform=valid_transforms(cfg)
                         )
    
    print(len(dataset))
    
    targets = dataset[27]
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
        path='./test_mask.png',
        dpi=300
        # show_img=True
    )

if __name__ == "__main__":
    main()
