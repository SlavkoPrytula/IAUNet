import cv2
import numpy as np
from os.path import join
import os.path as osp
import cv2

import sys
sys.path.append('.')

import albumentations as A

import torch
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from configs import cfg

from dataclasses import dataclass
from typing import Optional

@dataclass
class DataSample:
    image: torch.Tensor
    masks: Optional[torch.Tensor] = None
    bboxes: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    metadata: Optional[dict] = None


class COCOAnnotationParser:
    def __init__(self, coco):
        self.coco = coco

    def parse(self, anns, img_info, return_masks=True, return_bboxes=True, return_labels=True):
        h, w = img_info['height'], img_info['width']
        results = {}

        if return_masks:
            masks = np.zeros((h, w, len(anns)), dtype=np.uint8)
            for i, ann in enumerate(anns):
                masks[:, :, i] = self.coco.annToMask(ann)
            if masks.shape[-1] == 0:
                masks = np.zeros((h, w, 1), dtype=np.uint8)
            results['masks'] = masks

        if return_bboxes:
            bboxes = []
            for ann in anns:
                if 'bbox' in ann and ann['bbox']:
                    x, y, w_, h_ = ann['bbox']
                    bboxes.append([x, y, x + w_, y + h_])
                elif 'segmentation' in ann and ann['segmentation']:
                    xs, ys = [], []
                    for polygon in ann['segmentation']:
                        xs.extend(polygon[0::2])
                        ys.extend(polygon[1::2])
                    if xs and ys:
                        x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
                        bboxes.append([x_min, y_min, x_max, y_max])
                    else:
                        bboxes.append([0, 0, 0, 0])
                else:
                    bboxes.append([0, 0, 0, 0])
            results['bboxes'] = np.array(bboxes, dtype=np.float32)

        if return_labels:
            results['labels'] = np.array([ann['category_id'] - 1 for ann in anns], dtype=np.int64)

        return results
    

class BaseCOCODataset(Dataset):
    def __init__(self, cfg: cfg, dataset_type="train", transform=None, 
                 return_masks=True, return_bboxes=True, return_labels=True):
        if dataset_type == "train":
            self.img_folder = cfg.dataset.train_dataset.images
            self.ann_file = cfg.dataset.train_dataset.ann_file
        elif dataset_type == "valid":
            self.img_folder = cfg.dataset.valid_dataset.images
            self.ann_file = cfg.dataset.valid_dataset.ann_file
        elif dataset_type == "eval":
            self.img_folder = cfg.dataset.eval_dataset.images
            self.ann_file = cfg.dataset.eval_dataset.ann_file
        elif dataset_type == "occ":
            raise NotImplementedError

        self.cfg = cfg
        self.coco = COCO(join(self.ann_file))
        self.parser = COCOAnnotationParser(self.coco)
        self.image_ids = self.coco.getImgIds()
        self.transform = transform
        self.return_masks = return_masks
        self.return_bboxes = return_bboxes
        self.return_labels = return_labels
        
        self.mean = np.array(cfg.dataset.mean).reshape(1, 1, 3)
        self.std = np.array(cfg.dataset.std).reshape(1, 1, 3)
        self.total_size = len(self.image_ids)

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        annIds = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(annIds)

        image = self.get_image(img_info)
        image = (image - self.mean) / self.std

        parsed = self.parser.parse(
            anns, img_info,
            return_masks=self.return_masks,
            return_bboxes=self.return_bboxes,
            return_labels=self.return_labels
        )

        masks = parsed.get('masks')
        bboxes = parsed.get('bboxes')
        labels = parsed.get('labels')

        if self.transform and masks is not None:
            data = self.transform(image=image, mask=masks)
            image = data['image']
            masks = data['mask']

        keep = None
        if masks is not None:
            masks, keep = self.filter_empty_masks(masks, return_idx=True)
            masks = torch.tensor(masks.transpose((2, 0, 1)), dtype=torch.float32)
        image = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32)

        if labels is not None and keep is not None:
            labels = torch.tensor(labels, dtype=torch.int64)[keep]
        elif labels is not None:
            labels = torch.tensor(labels, dtype=torch.int64)

        if bboxes is not None and keep is not None:
            bboxes = bboxes[keep]
        if bboxes is not None:
            h, w = image.shape[-2:]
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            bboxes = box_xyxy_to_cxcywh(bboxes) / torch.tensor([w, h, w, h], dtype=torch.float32)

        metadata = self.img_infos(img_info, img_id)
        
        datasample = {
            "image": image, # (H, W)
            "instance_masks": masks, # (N, H, W)
            "labels": labels, # (N)
            "bboxes": bboxes, # (N, 4)
        }
        metadata = self.img_infos(idx)
        datasample.update(metadata)

        return datasample

    def get_image(self, img_info):
        img_path = join(self.img_folder, img_info['file_name'])
        image = cv2.imread(img_path, -1).astype(np.float32)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        if len(image.shape) != 3:
            image = np.repeat(image[..., np.newaxis], 3, axis=-1)
        return image

    def img_infos(self, img_info, img_id):
        img_path = join(self.img_folder, img_info['file_name'])
        fname, _ = osp.splitext(osp.basename(img_path))
        return {
            "img_id": img_id,
            "img_path": img_path,
            "ori_shape": [img_info["height"], img_info["width"]],
            "file_name": fname,
            "coco_id": img_id,
        }

    @staticmethod
    def filter_empty_masks(sample, return_idx=False):
        is_empty = np.all(sample == 0, axis=(0, 1))
        kept_indices = np.where(~is_empty)[0]
        sample = sample[..., kept_indices]
        if sample.shape[-1] == 0:
            sample = np.zeros(sample.shape[:-1] + (1,), dtype=sample.dtype)
        if return_idx:
            return sample, kept_indices
        else:
            return sample
        


# class BaseCOCODataset(Dataset):
#     def __init__(self, cfg: cfg, dataset_type="train", normalization=None, transform=None):
#         if dataset_type == "train":
#             self.img_folder = cfg.dataset.train_dataset.images
#             self.ann_file = cfg.dataset.train_dataset.ann_file
#         elif dataset_type == "valid":
#             self.img_folder = cfg.dataset.valid_dataset.images
#             self.ann_file = cfg.dataset.valid_dataset.ann_file
#         elif dataset_type == "eval":
#             self.img_folder = cfg.dataset.eval_dataset.images
#             self.ann_file = cfg.dataset.eval_dataset.ann_file
#         elif dataset_type == "occ":
#             raise NotImplementedError
        
#         self.cfg = cfg
#         self.coco = COCO(join(self.ann_file))
#         self.image_ids = self.coco.getImgIds()
#         self.dataset_type = dataset_type
#         self.normalization = normalization
#         self.transform = transform

        # self.mean = np.array(cfg.dataset.mean).reshape(1, 1, 3)
        # self.std = np.array(cfg.dataset.std).reshape(1, 1, 3)

#         self.total_size = len(self.image_ids)

#     def __len__(self):
#         return self.total_size

#     def __getitem__(self, idx):
#         image = self.get_image(idx)
#         masks = self.get_mask(idx)
#         labels = self.get_labels(idx)
#         bboxes = self.get_bboxes(idx)

#         image = (image - self.mean) / self.std

#         if self.transform:
#             data = self.transform(
#                 image=image, 
#                 mask=masks, 
#                 )
#             image = data['image']
#             masks = data['mask']

#         # (H, W, M) -> (H, W, N)
#         masks, keep = self.filter_empty_masks(masks, return_idx=True) 
    
#         image = image.transpose((2, 0, 1))
#         image = torch.tensor(image, dtype=torch.float32)
        
#         masks = masks.transpose((2, 0, 1))
#         masks = torch.tensor(masks, dtype=torch.float32)

#         labels = torch.tensor(labels, dtype=torch.int64)
#         labels = labels[keep]

#         h, w = image.shape[-2:]
#         bboxes = bboxes[keep]
#         bboxes = torch.tensor(bboxes, dtype=torch.float32)
#         bboxes = box_xyxy_to_cxcywh(bboxes) / torch.tensor([w, h, w, h], dtype=torch.float32)

#         target = {
#             "image": image, # (H, W)
#             "instance_masks": masks, # (N, H, W)
#             "labels": labels, # (N)
#             "bboxes": bboxes, # (N, 4)
#         }
#         metadata = self.img_infos(idx)
#         target.update(metadata)

#         return target
    

#     def get_mask(self, img_id: int, cat_id: list=[], iscrowd=None):
#         img_id = self.image_ids[img_id]
#         img_info = self.coco.loadImgs([img_id])[0]
#         annIds = self.coco.getAnnIds(imgIds=[img_id], catIds=cat_id, iscrowd=iscrowd)
#         anns = self.coco.loadAnns(annIds)
        
#         h, w = img_info['height'], img_info['width']
#         masks = np.zeros((h, w, len(anns)), dtype=np.uint8)
#         for i, ann in enumerate(anns):
#             mask = self.coco.annToMask(ann)
#             masks[:, :, i] = mask
            
#         if masks.shape[-1] == 0:
#             masks = np.zeros((h, w, 1), dtype=masks.dtype)
        
#         return masks


#     def get_image(self, img_id):
#         img_id = self.image_ids[img_id]
#         img_info = self.coco.loadImgs([img_id])[0]
#         img_path = join(self.img_folder, img_info['file_name'])
#         image = cv2.imread(img_path, -1).astype(np.float32)

#         if image is None:
#             raise FileNotFoundError(f"Image with id {img_id} not found in path: {img_path}")
        
#         if len(image.shape) != 3:
#             image = np.repeat(image[..., np.newaxis], 3, axis=-1)

#         return image


#     def img_infos(self, img_id):
#         img_id = self.image_ids[img_id]
#         metadata = {}

#         img_info = self.coco.loadImgs([img_id])[0]
#         img_path = join(self.img_folder, img_info['file_name'])
#         fname, name = osp.splitext(osp.basename(img_path))

#         metadata["img_id"] = img_id
#         metadata["img_path"] = img_path
#         metadata["ori_shape"] = [img_info["height"], img_info["width"]]
#         metadata["file_name"] = fname
#         metadata["coco_id"] = img_id

#         return metadata
    

#     def get_labels(self, img_id: int, cat_id: list=[], iscrowd=None):
#         img_id = self.image_ids[img_id]
#         annIds = self.coco.getAnnIds(imgIds=[img_id], catIds=cat_id, iscrowd=iscrowd)
#         anns = self.coco.loadAnns(annIds)

#         labels = np.array([ann['category_id'] - 1 for ann in anns])
#         return labels
    

#     def get_bboxes(self, img_id: int, cat_id: list = [], iscrowd=None):
#         """
#         Extract bounding boxes for all annotations of a given image.
#         Returns bboxes in xyxy format by default, or xywh if return_xywh=True.
#         """
#         img_id = self.image_ids[img_id]
#         annIds = self.coco.getAnnIds(imgIds=[img_id], catIds=cat_id, iscrowd=iscrowd)
#         anns = self.coco.loadAnns(annIds)

#         bboxes = []
#         for ann in anns:
#             if 'bbox' in ann and ann['bbox']:
#                 x, y, w, h = ann['bbox']
#                 bboxes.append([x, y, x + w, y + h])
#             elif 'segmentation' in ann and ann['segmentation']:
#                 for polygon in ann['segmentation']:
#                     xs = polygon[0::2]
#                     ys = polygon[1::2]
#                     x_min = min(xs)
#                     y_min = min(ys)
#                     x_max = max(xs)
#                     y_max = max(ys)
#                     bboxes.append([x_min, y_min, x_max, y_max])

#         bboxes = np.array(bboxes, dtype=np.float32)
#         return bboxes
    
    
#     @staticmethod
#     def filter_empty_masks(sample, return_idx=False):
#         # Compute a mask indicating whether each channel is empty
#         is_empty = np.all(sample == 0, axis=(0, 1))
#         kept_indices = np.where(~is_empty)[0]

#         sample = sample[..., kept_indices]

#         if sample.shape[-1] == 0:
#             # If all channels were empty, add an all-zero channel
#             sample = np.zeros(sample.shape[:-1] + (1,), dtype=sample.dtype)

#         if return_idx:
#             return sample, kept_indices
#         else:
#             return sample