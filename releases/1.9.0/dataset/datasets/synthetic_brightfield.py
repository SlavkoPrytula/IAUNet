import cv2
import numpy as np
import pandas as pd
import json
from os.path import join
import re

import cv2
import scipy.ndimage as ndi
from scipy.ndimage import binary_dilation, binary_erosion
from tifffile import tifffile

import sys
sys.path.append('.')

import torch
from torch.utils.data import Dataset

from utils.utils import flatten_mask
from pycocotools.coco import COCO
from configs import cfg



class SyntheticBrightfield_Dataset(Dataset):
    # _data_path = "/gpfs/space/home/prytula/data/datasets/cytoplasm_segmentation/synthetic_brightfield/[1024x1024]_[bf]_[not_normalized]_[aug4_scale]_[29.05.23]/masks/masks.npy"

    def __init__(self, df, run_type, img_size, normalization=None, transform=None):
        self.masks = np.load(f'{cfg.dataset.masks}/masks.npy', allow_pickle=True)
        self.df = df
        self.run_type = run_type
        self.img_size = img_size
        self.normalization = normalization
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.get_brightfield(idx)
        mask = self.get_cyto_mask(idx)
        
        # if self.normalization:
        #     image = self.normalization(image)

        if self.transform:
            data = self.transform(image=image, mask=mask)
            image = data['image']
            mask = data['mask']
            
        mask = self.filter_empty_masks(mask)
        # occluder = self.get_occluder(mask)
        # overlap = self.get_overlap(mask)

        # mask_bound = self.get_border(mask)
        # occluder_bound = self.get_border(occluder)


        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        
        mask = np.transpose(mask, (2, 0, 1))
        mask = torch.tensor(mask, dtype=torch.float32)

        # overlap = np.transpose(overlap, (2, 0, 1))
        # overlap = torch.tensor(overlap, dtype=torch.float32)
        
        # occluder = np.transpose(occluder, (2, 0, 1))
        # occluder = torch.tensor(occluder, dtype=torch.float32)

        # mask_bound = np.transpose(mask_bound, (2, 0, 1))
        # mask_bound = torch.tensor(mask_bound, dtype=torch.float32)

        # occluder_bound = np.transpose(occluder_bound, (2, 0, 1))
        # occluder_bound = torch.tensor(occluder_bound, dtype=torch.float32)
        
        # labels
        N, _, _ = mask.shape
        labels = torch.zeros(N, dtype=torch.int64)

        # zero_mask_ids = [i for i, m in enumerate(occluder) if np.all(m == 0)]
        # labels_occluders = torch.zeros(N, dtype=torch.int64)
        # labels_occluders[zero_mask_ids] = 1

        # target = {
        #     "image": image,
        #     "masks": {
        #         "instance": mask,
        #         "occluder": occluder,
        #     },
        #     "labels": {
        #         "instnace": labels
        #     }
        # }
        # tgt = targets["masks"][<name>]
        # src = outputs["masks"][pred_<name>]

        target = {
            "image": image,
            "masks": mask,
            # "overlaps": overlap,
            # "occluders": occluder,
            "labels": labels,
            # "labels_occluders": labels_occluders,
            # "masks_bounds": mask_bound,
            # "occluders_bounds": occluder_bound
        }

        return target
    
    
    # def get_overlap(self, img_id: int):
    #     mask = self.get_cyto_mask(img_id)
    #     mask = flatten_mask(mask, -1)
    #     mask[mask < 2] = 0
    #     mask[mask >= 2] = 1
        
    #     return mask

    def get_overlap(self, masks):
        H, W, N = masks.shape
        overlap_masks = np.zeros((H, W, N), dtype=np.uint8)
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    overlap = np.logical_and(masks[:, :, i], masks[:, :, j])
                    overlap_masks[:, :, i] = np.logical_or(overlap_masks[:, :, i], overlap)
        
        return overlap_masks
    

    def get_occluder(self, masks):
        # full occluder mask
        H, W, N = masks.shape
        aggregated_masks = np.zeros((H, W, N), dtype=np.uint8)

        for i in range(N):
            for j in range(N):
                if i != j and np.any(np.logical_and(masks[:, :, i], masks[:, :, j])):
                    aggregated_masks[:, :, i] = np.logical_or(aggregated_masks[:, :, i], masks[:, :, j])

        return aggregated_masks


    # def get_occluder(self, masks, distance=5):
    #     H, W, N = masks.shape
    #     aggregated_masks = np.zeros((H, W, N), dtype=np.uint8)
        
    #     dilated_mask_cache = {}
        
    #     for i in range(N):
    #         # print(i)
    #         mask_i = masks[:, :, i]
            
    #         if i not in dilated_mask_cache:
    #             dilated_mask_i = binary_dilation(mask_i, structure=np.ones((distance, distance)))
    #             dilated_mask_cache[i] = dilated_mask_i
    #         else:
    #             dilated_mask_i = dilated_mask_cache[i]
            
    #         for j in range(N):
    #             mask_j = masks[:, :, j]
    #             if i != j and np.any(np.logical_and(dilated_mask_i, mask_j)):
    #                 aggregated_masks[:, :, i] = np.logical_or(aggregated_masks[:, :, i], mask_j)
        
    #     return aggregated_masks

    

    def get_border(self, masks, width=16):
        H, W, N = masks.shape
        border_masks = np.zeros((H, W, N), dtype=np.uint8)

        for i in range(N):
            mask = masks[:, :, i]
            # Perform binary dilation to get border regions
            border_mask = binary_dilation(mask, structure=np.ones((width, width)))
            # Exclude the original mask region
            border_mask = np.logical_and(border_mask, np.logical_not(mask))
            border_masks[:, :, i] = border_mask

        return border_masks

        
    def get_brightfield(self, idx):
        image = tifffile.imread(f'{cfg.dataset.images}/syn_bf_{idx}.tiff')
        
        image += 1
        image /= 2.
        image = image.astype(np.float32)
        sampled_lambda = np.random.uniform(300, 700)
        image = np.random.poisson(image * sampled_lambda) / sampled_lambda

        return image
    

    def get_cyto_mask(self, idx):
        mask = self.masks[idx]
        return mask
    
    
    def get_border(self, mask, thickness=1):
        def disk_struct(radius):
            x = np.arange(-radius, radius + 1)
            x, y = np.meshgrid(x, x)
            r = x ** 2 + y ** 2
            struct = r < radius ** 2
            return struct
        
        struct = disk_struct(thickness+1)
        
        border = np.zeros(mask.shape)
        for i in range(mask.shape[-1]):
            obj_mask = mask[..., i]
            obj_erosed = ndi.binary_erosion(obj_mask, struct)
            obj_boundary = obj_mask - obj_erosed
            border[..., i] = obj_boundary
            
        return border
    
    
    @staticmethod
    def filter_empty_masks(sample):
        # filter empty channels
        _sample = []
        for i in range(sample.shape[-1]):
            if np.all(sample[..., i] == 0):
                continue
            _sample.append(sample[..., i])
            
        if not len(_sample):
            _sample = [np.zeros(sample.shape[:-1])]
        sample = np.stack(_sample, -1)

        return sample
    


df = pd.DataFrame({"cell_line": ["None"] * 100})
df.index = np.arange(0, len(df))
df['id'] = df.index


if __name__ == "__main__":
    from utils.visualise import visualize, visualize_grid_v2
    dataset = SyntheticBrightfield_Dataset(df, "valid", cfg.valid.size)
    targets = dataset[0]
    visualize(images=targets["image"][0, ...], path='./test_iamge.jpg', cmap='gray')

    print(targets["image"][0, ...].min(), targets["image"][0, ...].max())

    
    visualize_grid_v2(
        masks=targets["masks"], 
        path='./test_inst.jpg',
        ncols=5
    )
    # visualize_grid_v2(
    #     masks=targets["occluders"], 
    #     path='./test_occl.jpg',
    #     ncols=5
    # )
    # visualize_grid_v2(
    #     masks=targets["overlaps"], 
    #     path='./test_ovlp.jpg',
    #     ncols=5
    # )

    # visualize_grid_v2(
    #     masks=targets["masks_bounds"], 
    #     path='./test_inst_bound.jpg',
    #     ncols=5
    # )
    # visualize_grid_v2(
    #     masks=targets["occluders_bounds"], 
    #     path='./test_occl_bound.jpg',
    #     ncols=5
    # )
    