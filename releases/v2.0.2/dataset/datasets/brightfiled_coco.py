import cv2
import numpy as np
import pandas as pd
import json
from os.path import join
import re

import cv2
import scipy.ndimage as ndi
from scipy.ndimage import binary_dilation

import sys
sys.path.append('.')

import torch
from torch.utils.data import Dataset

from utils.utils import flatten_mask
from pycocotools.coco import COCO
from configs import cfg

from dataset.prepare_dataset import get_folds
from utils.registry import DATASETS

import torchvision.transforms as transforms
# from dataset.datasets import BaseCOCODataset
from dataset.datasets.base_coco_dataset import BaseCOCODataset
    

@DATASETS.register(name="brightfield_coco")
class BrightfieldCOCO(BaseCOCODataset):
    def __init__(self, cfg: cfg, dataset_type="train", normalization=None, transform=None):
        super().__init__(cfg, dataset_type, normalization, transform)

    def __getitem__(self, idx):
        # idx = self.image_ids[idx]
        image = self.get_image(idx)
        mask = self.get_mask(idx, cat_id=[1], iscrowd=0)
        labels = self.get_labels(idx, cat_id=[1], iscrowd=0)

        if idx not in self.means:
            self.means[idx] = np.mean(image, axis=None, keepdims=True)
            self.stds[idx] = np.std(image, axis=None, keepdims=True)

        if self.normalization:
            mean = self.means[idx]
            std = self.stds[idx]
            image = (image - mean) / std

        assert image.shape[-1] != 0
        assert mask.shape[-1] != 0

        if self.transform:
            data = self.transform(
                image=image, 
                mask=mask, 
                )
            image = data['image']
            mask = data['mask']

        # (H, W, M) -> (H, W, N)
        mask, keep = self.filter_empty_masks(mask, return_idx=True)   

        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        
        mask = np.transpose(mask, (2, 0, 1))
        mask = torch.tensor(mask, dtype=torch.float32)

        labels = torch.tensor(labels, dtype=torch.int64)
        labels -= 1

        # assert len(labels) == len(mask), f"{len(labels)}, {len(mask)}"

        # labels
        N, _, _ = mask.shape
        labels = torch.zeros(N, dtype=torch.int64)

        target = {
            "image": image,
            "masks": mask,
            "labels": labels,
        }
        metadata = self.img_infos(idx)
        target.update(metadata)

        return target
    


if __name__ == "__main__":
    from utils.visualise import visualize, visualize_grid_v2, plot3d
    from utils.normalize import normalize
    from utils.augmentations import train_transforms, valid_transforms
    from utils.registry import DATASETS_CFG
    import time

    time_s = time.time()
    
    cfg.dataset = DATASETS_CFG.get("brightfield_coco_v2.0")

    dataset = BrightfieldCOCO(cfg, dataset_type="train",
                      normalization=normalize,
                      transform=valid_transforms(cfg)
                      )
    
    print(len(dataset))

    targets = dataset[0]

    time_e = time.time()
    print(f'loaded in {time_e - time_s}(s)')

    print(targets["image"].shape)
    print(targets["labels"])

    print(targets["image"].shape, targets["masks"].shape)

    print(f'std: {targets["image"].std()}, mean: {targets["image"].mean()}')
    print(f'std: {targets["image"].std(dim=(1, 2))}, mean: {targets["image"].mean(dim=(1, 2))}')

    
    visualize(images=targets["image"][0, ...], path='./test_image.jpg', cmap='gray',)
    visualize(images=flatten_mask(targets["masks"].numpy(), axis=0)[0], path='./test_mask.jpg', cmap='gray',)
    
    