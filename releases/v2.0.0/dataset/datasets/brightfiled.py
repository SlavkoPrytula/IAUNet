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


def normalize2range(x, newRange=(0, 1)):
    xmin, xmax = np.min(x), np.max(x)
    norm = (x - xmin)/(xmax - xmin)
    
    if newRange == (0, 1):
        return(norm)
    elif newRange != (0, 1):
        return norm * (newRange[1] - newRange[0]) + newRange[0]
    

@DATASETS.register(name="brightfield")
class Brightfield_Dataset(Dataset):
    def __init__(self, cfg: cfg, is_train=True, normalization=None, transform=None):
        self.coco = COCO(join(cfg.dataset.coco_dataset))
        self.cfg = cfg
        self.is_train = is_train
        self.normalization = normalization
        self.transform = transform
        self.df = self.get_df()

        # self.prob_maps = np.load("/gpfs/space/home/prytula/data/mask_labels/x63_fl/np/overlap_segmentation/distance_maps/border_influence/prob_maps_[exp=0.7]_[norm=0.2-1]_[25.08.23].npy", allow_pickle=True)
        # self.occluder_cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.get_brightfield(idx)
        mask = self.get_cyto_mask(idx)
        # prob_map = self.get_prob_map(idx)
        
        # image = (image - np.amin(image)) / (np.amax(image) - np.amin(image))
        # image /= 28000.
        # image = image.astype(np.float32)

        if self.normalization:
            image = self.normalization(image)

        if self.transform:
            data = self.transform(
                image=image, 
                mask=mask, 
                # prob_map=prob_map
                )
            image = data['image']
            mask = data['mask']
            # prob_map = data['prob_map']
        
        # (H, W, M) -> (H, W, N)
        mask, keep = self.filter_empty_masks(mask, return_idx=True)     
        occluder = self.get_occluder(mask)

        # prob_map = prob_map[..., keep]
        # prob_map = self.get_prob_map(mask)

        # filter corresponding masks to match instance masks (H, W, M) -> (H, W, N)
        # occluder = occluder[..., keep]
        # overlap = self.get_overlap(mask)

        # mask_bound = self.get_border(mask)
        # occluder_bound = self.get_border(occluder)

        # image = np.concatenate([image, occluder], axis=-1)


        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        
        mask = np.transpose(mask, (2, 0, 1))
        mask = torch.tensor(mask, dtype=torch.float32)

        # overlap = np.transpose(overlap, (2, 0, 1))
        # overlap = torch.tensor(overlap, dtype=torch.float32)
        
        occluder = np.transpose(occluder, (2, 0, 1))
        occluder = torch.tensor(occluder, dtype=torch.float32)

        # prob_map = np.transpose(prob_map, (2, 0, 1))
        # prob_map = torch.tensor(prob_map, dtype=torch.float32)

        # mask_bound = np.transpose(mask_bound, (2, 0, 1))
        # mask_bound = torch.tensor(mask_bound, dtype=torch.float32)

        # occluder_bound = np.transpose(occluder_bound, (2, 0, 1))
        # occluder_bound = torch.tensor(occluder_bound, dtype=torch.float32)
        
        # labels
        N, _, _ = mask.shape
        labels = torch.zeros(N, dtype=torch.int64)
        # labels += 1

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
            "occluders": occluder,
            "labels": labels,
            # "labels_occluders": labels_occluders,
            # "masks_bounds": mask_bound,
            # "occluders_bounds": occluder_bound
            # "prob_maps": prob_map
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
    
    
    # def get_occluder(self, masks, idx):
    #     if idx not in self.occluder_cache:
    #         # Compute and cache the occluder mask
    #         occluder_mask = self._get_occluder(masks)
    #         self.occluder_cache[idx] = occluder_mask
    #     return self.occluder_cache[idx]
    

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

        # H, W, N = masks.shape
        # aggregated_masks = np.zeros((H, W, N), dtype=np.uint8)

        # for i in range(N):
        #     # Exclude the current mask i when computing overlaps
        #     other_masks = np.delete(masks, i, axis=2)
            
        #     # Compute the overlap of mask i with all other masks
        #     overlap_mask = np.any(other_masks * masks[:, :, i:i+1], axis=2)
            
        #     # Assign the computed overlap mask to aggregated_masks
        #     aggregated_masks[:, :, i] = overlap_mask

        # return masks * aggregated_masks
    

    # def get_occluder(self, masks):
    #     # full occluder mask
    #     H, W, N = masks.shape
    #     aggregated_masks = np.zeros((H, W, N), dtype=np.uint8)

    #     for i in range(N):
    #         for j in range(N):
    #             if i != j and np.any(np.logical_and(masks[:, :, i], masks[:, :, j])):
    #                 aggregated_masks[:, :, i] = np.logical_or(aggregated_masks[:, :, i], masks[:, :, j])

    #     return aggregated_masks


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

    def get_prob_map(self, mask):
        H, W, N = mask.shape
        sigma = 0.1

        # Initialize an empty array to store glow masks
        glow_masks = np.zeros((H, W, N))

        # Calculate distance maps for each object
        for i in range(N):
            object_mask = mask[:, :, i]
            distance_map = ndi.distance_transform_edt(1 - object_mask)  # Invert the mask for distance transform
            distance_map /= np.max(distance_map)  # Normalize distance map

            # Define the glow intensity function (you can adjust this function)
            glow_intensity = np.exp(-distance_map**2 / (2 * sigma**2))  # Gaussian function
            
            glow_masks[:, :, i] = glow_intensity

        glow_masks = np.round(glow_masks, 4)
        glow_masks[glow_masks < 0.1] = 1
        glow_masks[glow_masks != 1] = 1 - glow_masks[glow_masks != 1]
        glow_masks[glow_masks < 0.5] = 0.5

        ovlp = self.get_overlap(mask)
        glow_masks -= 0.3
        glow_masks[ovlp==1] = 1

        return glow_masks

    # def get_prob_map(self, img_id: int):
    #     img_id = self.df.loc[img_id]['mask_id']
    #     prob_map = self.prob_maps[img_id]

    #     # prob_map[prob_map!=0] = normalize2range(prob_map[prob_map!=0], (0.2, 1))
    #     # prob_map = normalize2range(prob_map, (0.5, 1))
        
    #     conf_overlaps = prob_map.copy()

    #     prob_map[prob_map!=0] = normalize2range(prob_map[prob_map!=0], (0, 0.5))

    #     prob_map = 1 - prob_map
        
    #     prob_map[conf_overlaps==1] = 1

    #     return prob_map


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
        images = [img_bf_lo, img_bf_hi]
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
    
    
#     def get_overlap(self, mask):
#         mask = flatten_mask(mask, axis=-1)
#         overlap = np.zeros(mask.shape)
#         overlap[mask>1] = 1
        
#         return overlap
    
    
    # @staticmethod
    # def filter_empty_masks(sample):
    #     # filter empty channels
    #     _sample = []
    #     for i in range(sample.shape[-1]):
    #         if np.all(sample[..., i] == 0):
    #             continue
    #         _sample.append(sample[..., i])
            
    #     if not len(_sample):
    #         _sample = [np.zeros(sample.shape[:-1])]
    #     sample = np.stack(_sample, -1)

    #     return sample
    
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

        df = pd.read_csv(cfg.dataset.csv_dataset_dir)[:100]  # dummy beffer crop - some images were skipped when labeling 
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
    from utils.visualise import visualize, visualize_grid_v2, plot3d
    from utils.normalize import normalize
    from utils.augmentations import train_transforms, valid_transforms
    import time

    time_s = time.time()
    dataset = Brightfield_Dataset(cfg, is_train=True,
                                  normalization=normalize,
                                  transform=train_transforms(cfg)
                                  )
    targets = dataset[0]
    time_e = time.time()
    print(f'loaded in {time_e - time_s}(s)')

    print(targets["image"].shape)

    # visualize(images=targets["image"][0, ...], path='./test_mask.jpg', cmap='plasma',)
    
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
    