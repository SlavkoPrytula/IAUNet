import cv2
import numpy as np
from os.path import join
import re

import tifffile
import cv2
import scipy.ndimage as ndi
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from utils.utils import flatten_mask
from .brightfiled import df
from pycocotools.coco import COCO
from configs import cfg



class Brightfield_Nuc_Dataset(Dataset):
    def __init__(self, df, run_type, img_size, normalization, transform):
        self.coco = COCO(join(cfg.dataset.coco_dataset))
        self.df = df
        self.run_type = run_type
        self.img_size = img_size
        self.normalization = normalization
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        bf_image = self.get_brightfield(idx)
        cyto_mask = self.get_cyto_mask(idx)
        nuc_mask = self.get_nuc_mask(idx)
        
        if self.normalization:
            bf_image = self.normalization(bf_image)

        if self.transform:
            data = self.transform(image=bf_image, mask=cyto_mask, mask1=nuc_mask)
            bf_image = data['image']
            cyto_mask = data['mask']
            nuc_mask = data['mask1']

        cyto_mask = self.filter_empty_masks(cyto_mask)
        nuc_mask = self.filter_empty_masks(nuc_mask)
    

        bf_image = np.transpose(bf_image, (2, 0, 1))
        bf_image = torch.tensor(bf_image, dtype=torch.float32)
        
        cyto_mask = np.transpose(cyto_mask, (2, 0, 1))
        cyto_mask = torch.tensor(cyto_mask, dtype=torch.float32)

        nuc_mask = np.transpose(nuc_mask, (2, 0, 1))
        nuc_mask = torch.tensor(nuc_mask, dtype=torch.float32)

        return bf_image, cyto_mask, cyto_mask
    

        
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
    
    
    def get_overlap(self, mask):
        mask = flatten_mask(mask, axis=-1)
        overlap = np.zeros(mask.shape)
        overlap[mask>1] = 1
        
        return overlap
    
    
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
    