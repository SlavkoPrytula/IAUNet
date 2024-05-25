import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

from pycocotools.coco import COCO
from os.path import join
from scipy.ndimage import binary_dilation

import sys
sys.path.append('.')

from utils.utils import flatten_mask
from dataset.prepare_dataset import get_folds
from utils.registry import DATASETS

from configs import cfg


@DATASETS.register(name="rectangle")
class Rectangle_Dataset(Dataset):
    def __init__(self, cfg: cfg, is_train=True, normalization=None, transform=None):
        self.coco = COCO(join(cfg.dataset.coco_dataset))
        self.cfg = cfg
        self.is_train = is_train
        self.normalization = normalization
        self.transform = transform
        self.df = self.get_df()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        mask = self.get_mask(idx)
        mask = np.transpose(mask, (1, 2, 0))
        
        image = flatten_mask(mask, -1)
        image[image > 2] = 2
        
        # if self.normalization:
        #     image = self.normalization(image)

        if self.transform:
            data = self.transform(image=image, mask=mask)
            image = data['image']
            mask = data['mask']
            
        mask = self.filter_empty_masks(mask)
        # overlap = self.get_overlap(mask)
        occluder = self.get_occluder(mask)

        # mask_bound = self.get_border(mask)
        # occluder_bound = self.get_border(occluder)

        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        
        mask = np.transpose(mask, (2, 0, 1))
        mask = torch.tensor(mask, dtype=torch.float32)

        # overlap = np.transpose(overlap, (2, 0, 1))
        # overlap = torch.tensor(overlap, dtype=torch.float32)

        occluder = np.transpose(occluder, (2, 0, 1))
        occluder = torch.tensor(occluder, dtype=torch.float32)

        # mask_bound = np.transpose(mask_bound, (2, 0, 1))
        # mask_bound = torch.tensor(mask_bound, dtype=torch.float32)

        # occluder_bound = np.transpose(occluder_bound, (2, 0, 1))
        # occluder_bound = torch.tensor(occluder_bound, dtype=torch.float32)

        # labels
        N, _, _ = mask.shape
        labels = torch.zeros(N, dtype=torch.int64)

        target = {
            "image": image,
            "masks": mask,
            # "overlaps": overlap,
            "occluders": occluder,
            "labels": labels,
            # "masks_bounds": mask_bound,
            # "occluders_bounds": occluder_bound
        }
        
        return target
    
    
    # def get_overlap(self, mask):
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


    
    # def get_occluder(self, masks):
    #     # exclude regions where occluder and occludee overlap
    #     H, W, N = masks.shape
    #     aggregated_masks = np.zeros((H, W, N), dtype=np.uint8)

    #     for i in range(N):
    #         for j in range(N):
    #             if i != j and np.any(np.logical_and(masks[:, :, i], masks[:, :, j])):
    #                 aggregated_masks[:, :, i] = np.logical_or(aggregated_masks[:, :, i], masks[:, :, j])

    #     # Exclude regions where masks overlap
    #     aggregated_masks = np.logical_and(aggregated_masks, np.logical_not(masks))

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


    def get_mask(self, img_id: int):
        img_id = self.df.loc[img_id]['id'] # handling correct fold indexing
        annIds = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(annIds)
        get_mask = lambda idx: self.coco.annToMask(anns[idx])

        mask = np.zeros((len(anns), 512, 512))
        for i in range(len(anns)):
            _mask = get_mask(i)
            mask[i] = _mask

        return mask 
    
    
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
    

    def get_df(self):
        df = pd.DataFrame({"cell_line": ["None"] * 100})
        df.index = np.arange(0, len(df))
        df['id'] = df.index

        # 5-fold split
        df = get_folds(self.cfg, df)

        fold = 0
        if self.is_train:
            df = df.query("fold!=@fold").reset_index(drop=True)
        else:
            df = df.query("fold==@fold").reset_index(drop=True)

        if self.cfg.verbose:
            print(df.groupby(['fold', 'cell_line'])['id'].count())

        return df
        



if __name__ == "__main__":
    from configs import cfg
    from utils.visualise import visualize, visualize_grid_v2
    from utils.normalize import normalize
    from utils.augmentations import train_transforms, valid_transforms

    dataset = Rectangle_Dataset(cfg, "train", normalization=normalize, transform=train_transforms(cfg))
    targets = dataset[2]
    visualize(images=targets["image"][0, ...], path='./test_mask.jpg', cmap='plasma')

    print(targets["image"][0, ...].shape)
    print(targets["image"][0, ...].min(), targets["image"][0, ...].max())

    
    visualize_grid_v2(
        masks=targets["masks"], 
        path='./test_inst.jpg',
        ncols=5
    )
    visualize_grid_v2(
        masks=targets["occluders"], 
        path='./test_occl.jpg',
        ncols=5
    )

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
    
    