import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

from pycocotools.coco import COCO
from os.path import join

from utils.utils import flatten_mask
from configs import cfg


df = pd.DataFrame({"cell_line": ["None"] * 100})
df.index = np.arange(0, len(df))
df['id'] = df.index



class Rectangle_Dataset(Dataset):
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
        overlap = self.get_overlap(mask)

        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        
        mask = np.transpose(mask, (2, 0, 1))
        mask = torch.tensor(mask, dtype=torch.float32)

        overlap = np.transpose(overlap, (2, 0, 1))
        overlap = torch.tensor(overlap, dtype=torch.float32)

        # labels
        N, _, _ = mask.shape
        labels = torch.zeros(N, dtype=torch.int64)

        target = {
            "image": image,
            "masks": mask,
            "overlaps": overlap,
            "labels": labels
        }
        
        return target
    
    
    def get_overlap(self, mask):
        mask = flatten_mask(mask, -1)
        mask[mask < 2] = 0
        mask[mask >= 2] = 1
        
        return mask


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
    