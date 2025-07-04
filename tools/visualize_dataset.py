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
                      transform=get_valid_transforms(cfg), 
                      return_bboxes=False)

    print(f"Loaded dataset: {cfg.dataset.name}, length: {len(dataset)}")
    sample = dataset[2]

    print(f"Sample image shape: {sample['image'].shape}")
    print(f"Sample masks shape: {sample['instance_masks'].shape}")
    print(f"Sample resized shape: {sample['resized_shape']}")
    print(f"Sample ori shape: {sample['ori_shape']}")

    print(f"Sample keys: {list(sample.keys())}")


    # from evaluation.evaluators.coco_evaluator import remove_padding
    # sample["image"] = remove_padding(
    #     sample["image"], 
    #     img_size=sample["resized_shape"],
    #     output_height=sample["ori_shape"][0],
    #     output_width=sample["ori_shape"][1],
    #     rescale=True
    # )

    # sample["instance_masks"] = remove_padding(
    #     sample["instance_masks"], 
    #     img_size=sample["resized_shape"],
    #     output_height=sample["ori_shape"][0],
    #     output_width=sample["ori_shape"][1],
    #     rescale=True
    # )

    print(f"Sample image shape: {sample['image'].shape}")
    print(f"Sample mask shape: {sample['instance_masks'].shape}")
    visualize(images=sample["image"][0, ...], path="./test_image.jpg", cmap="gray", show_title=False)
    
    H, W = sample["image"].shape[-2:]
    # visualize_masks(
    #     figsize=[30, 30],
    #     img=sample["image"][0, ...],
    #     masks=sample["instance_masks"],
    #     bboxes=sample["bboxes"],
    #     shape=[H, W],
    #     alpha=0.65,
    #     border_size=10,
    #     bbox_linewidth=10,
    #     draw_border=True,
    #     border_color="white",
    #     static_color=True,
    #     path="./test_mask.png",
    #     dpi=100
    # )

    # save_coco_vis(
    #     img=sample["image"][0, ...],
    #     gt_coco=COCO(cfg.dataset.train_dataset.ann_file),
    #     pred_coco=COCO(cfg.dataset.train_dataset.ann_file),
    #     idx=sample["coco_id"],
    #     shape=sample["ori_shape"],
    #     alpha=0.65,
    #     draw_border=True,
    #     border_size=4,
    #     path="./test_mask_coco.png",
    # )

if __name__ == "__main__":
    main()