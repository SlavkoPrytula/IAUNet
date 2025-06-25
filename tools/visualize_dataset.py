import sys 
sys.path.append("./")

import hydra
from pycocotools.coco import COCO
from configs import cfg
from dataset import *
from utils.registry import DATASETS
from utils.augmentations import get_train_transforms, get_valid_transforms
from visualizations import visualize, visualize_masks, save_coco_vis


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: cfg):
    dataset = DATASETS.get(cfg.dataset.type)
    dataset = dataset(cfg, 
                      dataset_type="train", 
                      transform=get_valid_transforms(cfg))

    print(f"Loaded dataset: {cfg.dataset.name}, length: {len(dataset)}")
    sample = dataset[0]

    print(f"Sample keys: {list(sample.keys())}")

    visualize(images=sample["image"][0, ...], path="./test_image.jpg", cmap="gray", show_title=False)
    
    H, W = sample["image"].shape[-2:]
    visualize_masks(
        figsize=[30, 30],
        img=sample["image"][0, ...],
        masks=sample["instance_masks"],
        bboxes=sample["bboxes"],
        shape=[H, W],
        alpha=0.65,
        border_size=10,
        bbox_linewidth=10,
        draw_border=True,
        static_color=False,
        path="./test_mask.png",
        dpi=100
    )

    # save_coco_vis(
    #     img=sample["image"][0, ...],
    #     gt_coco=COCO(cfg.dataset.train_dataset.ann_file),
    #     pred_coco=COCO(cfg.dataset.train_dataset.ann_file),
    #     idx=1,
    #     shape=sample["ori_shape"],
    #     alpha=0.65,
    #     draw_border=True,
    #     border_size=5,
    #     path="./test_mask_coco.png",
    # )

if __name__ == "__main__":
    main()