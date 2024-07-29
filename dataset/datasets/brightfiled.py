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


def normalize2range(x, newRange=(0, 1)):
    xmin, xmax = np.min(x), np.max(x)
    norm = (x - xmin)/(xmax - xmin)
    
    if newRange == (0, 1):
        return(norm)
    elif newRange != (0, 1):
        return norm * (newRange[1] - newRange[0]) + newRange[0]
    

@DATASETS.register(name="brightfield")
class Brightfield_Dataset(Dataset):
    def __init__(self, cfg: cfg, dataset_type="train", normalization=None, transform=None):
        self.coco = COCO(join(cfg.dataset.coco_dataset))
        self.cfg = cfg
        self.dataset_type = dataset_type
        self.normalization = normalization
        self.transform = transform
        self.df = self.get_df()

        self.means = {}
        self.stds = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.get_brightfield(idx)
        mask = self.get_cyto_mask(idx)
        # self.get_phase_contrast(idx)

        if idx not in self.means:
            self.means[idx] = np.mean(image, axis=None, keepdims=True)
            self.stds[idx] = np.std(image, axis=None, keepdims=True)
        
        if self.normalization:
            mean = self.means[idx]
            std = self.stds[idx]
            image = (image - mean) / std


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
    
    def img_infos(self, img_id):
        img_id = int(self.df['id'].values[img_id])
        metadata = {}

        metadata["img_id"] = img_id
        metadata["img_path"] = f"{img_id}.png"
        metadata["ori_shape"] = [512, 512]

        return metadata
    

    def get_overlap(self, masks):
        H, W, N = masks.shape
        overlap_masks = np.zeros((H, W, N), dtype=np.uint8)
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    overlap = np.logical_and(masks[:, :, i], masks[:, :, j])
                    overlap_masks[:, :, i] = np.logical_or(overlap_masks[:, :, i], overlap)
        
        return overlap_masks
    

    # @lru_cache(maxsize=None)
    def get_occluder(self, masks):
        # full occluder mask
        H, W, N = masks.shape
        aggregated_masks = np.zeros((H, W, N), dtype=np.uint8)

        for i in range(N):
            for j in range(N):
                if i != j and np.any(np.logical_and(masks[:, :, i], masks[:, :, j])):
                    aggregated_masks[:, :, i] = np.logical_or(aggregated_masks[:, :, i], masks[:, :, j])

        return aggregated_masks


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

        
    def phase_contrast(self, image_name: str):
        # phase-contrast image
        pc_path = join(cfg.dataset.dataset_x63_dir, f'{image_name}_phase.png')
        print(pc_path)
        img_pc = cv2.imread(pc_path, -1)

        return img_pc


    def brightfield(self, image_name):
        encoded_image_name = re.split('(\d+)', image_name.split('_')[0])
        img_metadata = self.df.loc[
            (self.df['Row'] == int(encoded_image_name[1])) & 
            (self.df['Col'] == int(encoded_image_name[3])) & 
            (self.df['FieldID'] == int(encoded_image_name[5]))
        ]

        bf_lo_path = img_metadata['bf_lower'].values[0]
        bf_hi_path = img_metadata['bf_higher'].values[0]

        # brightfield lower
        img_bf_lo = cv2.imread(bf_lo_path, -1)

        # brightfield higher
        img_bf_hi = cv2.imread(bf_hi_path, -1)

        return img_bf_hi, img_bf_lo


    def fl_mask(self, image_name: str):
        encoded_image_name = re.split('(\d+)', image_name.split('_')[0])
        new_image_name = f'{encoded_image_name[1]}{encoded_image_name[3]}K1F{int(encoded_image_name[5])}P1R1.png'
        fl_path = join(cfg.dataset.fl_masks, new_image_name)

        img_fl = cv2.imread(fl_path, -1)

        img_fl = img_fl / img_fl.max()
        img_fl = np.ceil(img_fl) 
        
        return img_fl


    def get_phase_contrast(self, img_id: int):
        image_name = f"R{str(self.df.iloc[img_id]['Row']).zfill(2)}C{str(self.df.iloc[img_id]['Col']).zfill(2)}F{str(self.df.iloc[img_id]['FieldID']).zfill(2)}"

        img_pc = self.phase_contrast(image_name)
        images = [img_pc, img_pc, img_pc]
        images = np.stack(images, axis=-1).astype(np.float32)
        
        return images


    def get_brightfield(self, img_id: int):
        image_name = f"R{str(self.df.iloc[img_id]['Row']).zfill(2)}C{str(self.df.iloc[img_id]['Col']).zfill(2)}F{str(self.df.iloc[img_id]['FieldID']).zfill(2)}"

        img_bf_hi, img_bf_lo = self.brightfield(image_name)
        images = [img_bf_lo, img_bf_hi, img_bf_hi]
        # images = [img_bf_lo, img_bf_hi]
        images = np.stack(images, axis=-1).astype(np.float32)

        return images


    def get_cyto_mask(self, img_id: int, cat_id: list=[0], iscrowd=None):
        img_id = self.df.loc[img_id]['mask_id']
        annIds = self.coco.getAnnIds(imgIds=img_id, catIds=cat_id, iscrowd=iscrowd)
        anns = self.coco.loadAnns(annIds)
        get_mask = lambda idx: self.coco.annToMask(anns[idx])

        h, w = get_mask(0).shape
        mask = np.zeros((len(anns), 1080, 1080))
        for i in range(len(anns)):
            _mask = get_mask(i)
            _mask = cv2.resize(_mask, (1080, 1080))
            mask[i] = _mask
        mask = np.transpose(mask, (1, 2, 0))
    
        return mask
    
    
    def get_nuc_mask(self, img_id: int):
        image_name = self.df.iloc[img_id]['fl_name']
        encoded_image_name = re.split('(\d+)', image_name.split('_')[0])
        new_image_name = f'{encoded_image_name[1]}{encoded_image_name[3]}K1F{int(encoded_image_name[5])}P1R1.png'
        fl_path = join(cfg.dataset.fl_masks, new_image_name)

        fl_mask = cv2.imread(fl_path, -1)
        
        mask = np.zeros((len(np.unique(fl_mask)), 1080, 1080))
        for i in np.unique(fl_mask):
            _mask = fl_mask.copy()
            _mask[_mask != i] = 0
            _mask[_mask == i] = 1
            _mask = cv2.resize(_mask, (1080, 1080))
            mask[i] = _mask
        mask = np.transpose(mask, (1, 2, 0))
        mask = mask[..., 1:]
    
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
    def filter_empty_masks(sample, return_idx=False):
        # Compute a mask indicating whether each channel is empty
        is_empty = np.all(sample == 0, axis=(0, 1))
        kept_indices = np.where(~is_empty)[0]

        sample = sample[..., kept_indices]

        if sample.shape[-1] == 0:
            # If all channels were empty, add an all-zero channel
            sample = np.zeros(sample.shape[:-1] + (1,), dtype=sample.dtype)

        if return_idx:
            return sample, kept_indices
        else:
            return sample


    def get_df(self):
        _json = open(join(self.cfg.dataset.coco_dataset))
        json_data = json.load(_json)

        df = pd.read_csv(cfg.dataset.csv_dataset_dir)[:200]  # dummy beffer crop - some images were skipped when labeling 
        df.index = np.arange(0, len(df))
        df['mask_id'] = 0
        df['fl_name'] = 0

        # --------------------
        # Preprocess dataframe
        print(json_data['images'])
        for i in range(len(json_data['images'])):
            image_name = json_data['images'][i]['file_name'].split('-')[-1]
            image_name = image_name.split('_')[0]

            encoded_image_name = re.split('(\d+)', image_name.split('_')[0])
            df.loc[
                (df['Row'] == int(encoded_image_name[1])) & 
                (df['Col'] == int(encoded_image_name[3])) & 
                (df['FieldID'] == int(encoded_image_name[5])),
                'mask_id'
            ] = i + 1


            df.loc[
                (df['Row'] == int(encoded_image_name[1])) & 
                (df['Col'] == int(encoded_image_name[3])) & 
                (df['FieldID'] == int(encoded_image_name[5])),
                'fl_name'
            ] = image_name

        df = df.drop(df[df['mask_id'] == 0].index)
        df['mask_id'] -= 1

        df.index = np.arange(0, len(df))
        df['id'] = df.index
        
        # 5-fold split
        df = get_folds(self.cfg, df)

        fold = 0
        if self.dataset_type == "train":
            df = df.query("fold!=@fold").reset_index(drop=True)
        elif self.dataset_type in ["valid", "eval"]:
            df = df.query("fold==@fold").reset_index(drop=True)

        if self.cfg.verbose:
            print(df.groupby(['fold', 'cell_line'])['id'].count())

        return df
        



if __name__ == "__main__":
    from utils.visualise import visualize, visualize_grid_v2, plot3d
    from utils.normalize import normalize
    from utils.augmentations import train_transforms, valid_transforms
    import time
    from utils.registry import DATASETS_CFG


    cfg.dataset = "brightfield_v2.0"
    cfg.dataset = DATASETS_CFG.get(cfg.dataset)

    time_s = time.time()
    dataset = Brightfield_Dataset(cfg, dataset_type="train",
                                  normalization=normalize,
                                  transform=train_transforms(cfg)
                                  )
    targets = dataset[0]

    print(len(dataset))

    print(f'std: {targets["image"].std(dim=(1, 2))}, mean: {targets["image"].mean(dim=(1, 2))}')
    print()

    time_e = time.time()
    print(f'loaded in {time_e - time_s}(s)')

    print(targets["image"].shape)

    visualize(images=targets["image"][0, ...], 
              path='./test_image.jpg', cmap='gray',)
    visualize(images=flatten_mask(targets["masks"].numpy(), axis=0)[0], 
              path='./test_mask.jpg', cmap='gray',)


    
    # print(targets["image"][0, ...].shape)
    # print(targets["image"][0, ...].min(), targets["image"][0, ...].max())

    # visualize_grid_v2(
    #     masks=targets["masks"] * targets["prob_maps"], 
    #     path='./test_inst.jpg',
    #     ncols=5
    # )

    # plot3d(
    #     image=targets["prob_maps"][4],
    #     path='./test_prob_map_3d.jpg',
    #     cmap='plasma'
    # )

    # # visualize_grid_v2(
    # #     masks=targets["occluders"], 
    # #     path='./test_occl.jpg',
    # #     ncols=5
    # # )
    # visualize_grid_v2(
    #     masks=targets["prob_maps"], 
    #     path='./test_prob_map.jpg',
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
    