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

from utils.registry import DATASETS


@DATASETS.register(name="original_plus_synthetic_brightfield")
class OriginalPlusSyntheticBrightfield(Dataset):
    # def __init__(self, df, run_type, img_size, normalization=None, transform=None):
    def __init__(self, cfg: cfg, is_train=True, normalization=None, transform=None):
        self.coco = COCO(join(cfg.dataset.coco_dataset))
        self.synthetic_masks = np.load(f'{cfg.dataset.masks}/masks.npy', allow_pickle=True)
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
        
        if self.normalization:
            image = self.normalization(image)

        if self.transform:
            data = self.transform(image=image, mask=mask)
            image = data['image']
            mask = data['mask']
            
        mask = self.filter_empty_masks(mask)
        occluder = self.get_occluder(mask)


        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        
        mask = np.transpose(mask, (2, 0, 1))
        mask = torch.tensor(mask, dtype=torch.float32)

        occluder = np.transpose(occluder, (2, 0, 1))
        occluder = torch.tensor(occluder, dtype=torch.float32)
        
        # labels
        N, _, _ = mask.shape
        labels = torch.zeros(N, dtype=torch.int64)

        zero_mask_ids = [i for i, m in enumerate(occluder) if torch.all(m == 0)]
        labels_occluders = torch.zeros(N, dtype=torch.int64)
        labels_occluders[zero_mask_ids] = 1

        target = {
            "image": image,
            "masks": mask,
            "occluders": occluder,
            "labels": labels,
            "labels_occluders": labels_occluders,
        }

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

        
    def get_brightfield(self, img_id: int):
        mask_type = self.df.loc[img_id]['mask_type']
        
        if mask_type == 'original':
            image_name = f"\
                R{str(self.df.iloc[img_id]['Row']).zfill(2)} \
                C{str(self.df.iloc[img_id]['Col']).zfill(2)} \
                F{str(self.df.iloc[img_id]['FieldID']).zfill(2)}"

            img_bf_hi, img_bf_lo = self.brightfield(image_name)
            image = [img_bf_lo, img_bf_hi]
            image = np.stack(image, axis=-1).astype(np.float32)
        else:
            img_id = self.df.loc[img_id]['mask_id']
            image = tifffile.imread(f'{cfg.dataset.images}/syn_bf_{img_id}.tiff')
            
            image += 1
            image /= 2.
            image = image.astype(np.float32)
            sampled_lambda = np.random.uniform(100, 500)
            image = np.random.poisson(image * sampled_lambda) / sampled_lambda

        return image
    

    def get_cyto_mask(self, img_id: int, cat_id: list=[0], iscrowd=None):
        mask_type = self.df.loc[img_id]['mask_type']
        img_id = self.df.loc[img_id]['mask_id']

        if mask_type == 'original':
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
        else:
            mask = self.synthetic_masks[img_id]
        return mask
    
    
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
    


_json = open(join(cfg.dataset.coco_dataset))
json_data = json.load(_json)

df = pd.read_csv(cfg.dataset.csv_dataset_dir)[:100]  # dummy buffer crop - some images were skipped when labeling 
df.index = np.arange(0, len(df))
df['mask_id'] = 0
df['fl_name'] = 0

# --------------------
# Preprocess dataframe
for i in range(28):
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

# from dataset.prepare_dataset import get_folds
def _get_folds(cfg: cfg, df):
    from sklearn.model_selection import StratifiedGroupKFold
    skf = StratifiedGroupKFold(n_splits=cfg.train.n_folds, shuffle=True, random_state=cfg.seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['cell_line'], groups=df['id'])):
        df.loc[val_idx, 'fold'] = fold

    return df

df = _get_folds(cfg, df)   # original split
df['mask_type'] = 'original'

synt_df = pd.DataFrame({"cell_line": ["None"] * 100, "fold": 1, "mask_type": 'synthetic'})
# synt_df.index = np.arange(0, len(synt_df))
synt_df['mask_id'] = np.arange(0, len(synt_df))


df = df.append(synt_df)
df.index = np.arange(0, len(df))
df['id'] = df.index

df = df.fillna(0)
df['Row'] = df['Row'].astype(int)
df['Col'] = df['Col'].astype(int)
df['FieldID'] = df['FieldID'].astype(int)
# df['id'] = df['id'].astype(int)




if __name__ == "__main__":
    from utils.visualise import visualize, visualize_grid_v2
    from utils.normalize import normalize
    dataset = OriginalPlusSyntheticBrightfield(df, "valid", cfg.valid.size, normalization=normalize)
    targets = dataset[0]
    visualize(images=targets["image"][0, ...], path='./test_image.jpg', cmap='gray')

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
    