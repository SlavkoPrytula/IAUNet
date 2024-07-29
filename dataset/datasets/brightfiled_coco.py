import torch
import numpy as np

import cv2
import scipy.ndimage as ndi
from scipy.ndimage import binary_dilation

import albumentations as A

import sys
sys.path.append('.')

from utils.utils import flatten_mask
from configs import cfg

from utils.registry import DATASETS

from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from dataset.datasets.base_coco_dataset import BaseCOCODataset
    

# @DATASETS.register(name="brightfield_coco")
# class BrightfieldCOCO(BaseCOCODataset):
#     def __init__(self, cfg: cfg, dataset_type="train", normalization=None, transform=None):
#         super().__init__(cfg, dataset_type, normalization, transform)

@DATASETS.register(name="brightfield_coco")
class BrightfieldCOCO(BaseCOCODataset):
    def __init__(self, cfg: cfg, dataset_type="train", normalization=None, transform=None):
        super().__init__(cfg, dataset_type, normalization, transform)

    def __getitem__(self, idx):
        # NOTE: Experimental
        # if type(idx) == list or type(idx) == tuple:
        #     idx, _size = idx
        # else:
        #     idx, _size = idx, (512, 512)

        # idx = self.image_ids[idx]
        image = self.get_image(idx)
        mask = self.get_mask(idx)
        labels = self.get_labels(idx)

        if idx not in self.means:
            self.means[idx] = np.mean(image, axis=(0, 1), keepdims=True)
            self.stds[idx] = np.std(image, axis=(0, 1), keepdims=True)

        if self.normalization:
            mean = self.means[idx]
            std = self.stds[idx]
            image = (image - mean) / std

        assert image.shape[-1] != 0
        assert mask.shape[-1] != 0

        if self.transform:
            # data = A.Resize(*_size)(
            #     image=image, 
            #     mask=mask, 
            #     )
            # image = data['image']
            # mask = data['mask']

            data = self.transform(
                image=image, 
                mask=mask, 
                )
            image = data['image']
            mask = data['mask']

        # (H, W, M) -> (H, W, N)
        mask, keep = self.filter_empty_masks(mask, return_idx=True) 
        # occulder = self.get_actual_overlaps(mask)
        bboxes = self.masks_to_boxes(mask)
        labels = labels[keep]

        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        
        mask = np.transpose(mask, (2, 0, 1))
        mask = torch.tensor(mask, dtype=torch.float32)
        
        # occulder = np.transpose(occulder, (2, 0, 1))
        # occulder = torch.tensor(occulder, dtype=torch.float32)

        labels = torch.tensor(labels, dtype=torch.int64)
        # labels -= 1

        # labels
        # N, _, _ = mask.shape
        # labels = torch.zeros(N, dtype=torch.int64)

        # bboxes
        h, w = image.shape[-2:]
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        bboxes = box_xyxy_to_cxcywh(bboxes)
        bboxes = bboxes / torch.tensor([w, h, w, h], dtype=torch.float32)

        target = {
            "image": image,
            "masks": mask,
            # "occluders": occulder,
            "labels": labels,
            "bboxes": bboxes
        }
        metadata = self.img_infos(idx)
        target.update(metadata)

        return target
    
    
    def get_occluder(self, masks):
        # full occluder mask
        H, W, N = masks.shape
        aggregated_masks = np.zeros((H, W, N), dtype=np.uint8)

        for i in range(N):
            for j in range(N):
                if i != j and np.any(np.logical_and(masks[:, :, i], masks[:, :, j])):
                    aggregated_masks[:, :, i] = np.logical_or(aggregated_masks[:, :, i], masks[:, :, j])

        return aggregated_masks

    def get_actual_overlaps(self, masks):
        # Initialize an array to store the overlap regions for each mask
        H, W, N = masks.shape
        overlap_regions = np.zeros((H, W, N), dtype=np.uint8)

        for i in range(N):
            for j in range(N):
                if i != j:
                    overlap = np.logical_and(masks[:, :, i], masks[:, :, j])
                    
                    # Store the overlap in both corresponding regions
                    overlap_regions[:, :, i] = np.logical_or(overlap_regions[:, :, i], overlap)
                    overlap_regions[:, :, j] = np.logical_or(overlap_regions[:, :, j], overlap)

        return overlap_regions
    


if __name__ == "__main__":
    from utils.visualise import visualize, visualize_grid_v2, plot3d
    from visualizations import visualize_masks
    from utils.augmentations import normalize
    from utils.augmentations import train_transforms, valid_transforms
    from utils.registry import DATASETS_CFG
    import time

    time_s = time.time()
    
    cfg.dataset = DATASETS_CFG.get("brightfield_coco_v2.0")

    dataset = BrightfieldCOCO(cfg, dataset_type="valid",
                      normalization=normalize,
                      transform=valid_transforms(cfg)
                      )
    
    print(len(dataset))

    targets = dataset[5]

    time_e = time.time()
    print(f'loaded in {time_e - time_s}(s)')

    print(targets["image"].shape)
    print(targets["labels"])

    print(targets["image"].shape, targets["masks"].shape)

    print(f'std: {targets["image"].std()}, mean: {targets["image"].mean()}')
    print(f'std: {targets["image"].std(dim=(1, 2))}, mean: {targets["image"].mean(dim=(1, 2))}')

    
    # visualize(images=targets["image"][0, ...], path='./test_image.jpg', cmap='gray',)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(30, 30))
    plt.imshow(targets["image"][0, ...], cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(f"./test_image.jpg", bbox_inches='tight', pad_inches=0)
    plt.close()
    
    mask = flatten_mask(targets["masks"].numpy(), axis=0)[0]
    mask[mask > 1] = 1
    plt.figure(figsize=(30, 30))
    plt.imshow(mask, cmap='viridis')
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(f"./test_mask.jpg", bbox_inches='tight', pad_inches=0)
    plt.close()
    # visualize(images=mask, path='./test_mask.jpg', cmap='viridis',)
    # visualize(images=flatten_mask(targets["masks"].numpy(), axis=0)[0], path='./test_mask.jpg', cmap='gray',)

    
    # bbox.
    h, w = targets["image"].shape[-2:]
    bboxes = box_cxcywh_to_xyxy(targets["bboxes"])
    bboxes = bboxes * torch.tensor([w, h, w, h], dtype=torch.float32)

    # visualize_grid_v2(
    #     masks=targets["masks"].numpy(), 
    #     bboxes=bboxes.numpy(), 
    #     path='./test_inst.jpg',
    #     ncols=5
    # )

    # visualize_grid_v2(
    #     masks=targets["occluders"].numpy(), 
    #     path='./test_occl.jpg',
    #     ncols=5
    # )
    

    # H, W = targets["ori_shape"]
    # visualize_masks(
    #     img=targets["image"][0, ...],
    #     masks=targets["masks"],
    #     shape=[H, W],
    #     alpha=0.65,
    #     draw_border=True, 
    #     static_color=False,
    #     path='./test_mask.jpg'
    # )
    