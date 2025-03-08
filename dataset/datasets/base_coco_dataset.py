import cv2
import numpy as np
from os.path import join
import os.path as osp
import cv2
from PIL import Image

import sys
sys.path.append('.')

import albumentations as A

import torch
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from configs import cfg


class BaseCOCODataset(Dataset):
    def __init__(self, cfg: cfg, dataset_type="train", normalization=None, transform=None):
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
        self.image_ids = self.coco.getImgIds()
        self.dataset_type = dataset_type
        self.normalization = normalization
        self.transform = transform

        self.mean = np.array(cfg.dataset.mean).reshape(1, 1, 3)
        self.std = np.array(cfg.dataset.std).reshape(1, 1, 3)

        self.total_size = len(self.image_ids)

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        image = self.get_image(idx)
        masks = self.get_mask(idx)
        labels = self.get_labels(idx)

        image = (image - self.mean) / self.std

        if self.transform:
            data = self.transform(
                image=image, 
                mask=masks, 
                )
            image = data['image']
            masks = data['mask']

        # (H, W, M) -> (H, W, N)
        masks, keep = self.filter_empty_masks(masks, return_idx=True) 
        # bboxes = self.masks_to_boxes(masks)
    
        image = image.transpose((2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        
        masks = masks.transpose((2, 0, 1))
        masks = torch.tensor(masks, dtype=torch.float32)

        labels = torch.tensor(labels, dtype=torch.int64)
        labels = labels[keep]

        # h, w = image.shape[-2:]
        # bboxes = torch.tensor(bboxes, dtype=torch.float32)
        # bboxes = box_xyxy_to_cxcywh(bboxes) / torch.tensor([w, h, w, h], dtype=torch.float32)

        target = {
            "image": image, # (H, W)
            "instance_masks": masks, # (N, H, W)
            "labels": labels, # (N)
            # "bboxes": bboxes,
        }
        metadata = self.img_infos(idx)
        target.update(metadata)

        return target
    

    def get_mask(self, img_id: int, cat_id: list=[], iscrowd=None):
        img_id = self.image_ids[img_id]
        img_info = self.coco.loadImgs([img_id])[0]
        annIds = self.coco.getAnnIds(imgIds=[img_id], catIds=cat_id, iscrowd=iscrowd)
        anns = self.coco.loadAnns(annIds)
        
        h, w = img_info['height'], img_info['width']
        masks = np.zeros((h, w, len(anns)), dtype=np.uint8)
        for i, ann in enumerate(anns):
            mask = self.coco.annToMask(ann)
            masks[:, :, i] = mask
            
        if masks.shape[-1] == 0:
            masks = np.zeros((h, w, 1), dtype=masks.dtype)
        
        return masks


    def get_image(self, img_id):
        img_id = self.image_ids[img_id]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = join(self.img_folder, img_info['file_name'])
        image = cv2.imread(img_path, -1).astype(np.float32)

        # image = Image.open(img_path)
        # image = np.array(image, dtype=np.float32)

        if image is None:
            raise FileNotFoundError(f"Image with id {img_id} not found in path: {img_path}")
        
        if len(image.shape) != 3:
            image = np.repeat(image[..., np.newaxis], 3, axis=-1)

        return image


    def img_infos(self, img_id):
        img_id = self.image_ids[img_id]
        metadata = {}

        img_info = self.coco.loadImgs([img_id])[0]
        img_path = join(self.img_folder, img_info['file_name'])
        fname, name = osp.splitext(osp.basename(img_path))

        metadata["img_id"] = img_id
        metadata["img_path"] = img_path
        metadata["ori_shape"] = [img_info["height"], img_info["width"]]
        metadata["file_name"] = fname
        metadata["coco_id"] = img_id

        return metadata
    

    def get_labels(self, img_id: int, cat_id: list=[], iscrowd=None):
        img_id = self.image_ids[img_id]
        annIds = self.coco.getAnnIds(imgIds=[img_id], catIds=cat_id, iscrowd=iscrowd)
        anns = self.coco.loadAnns(annIds)

        labels = np.array([ann['category_id'] - 1 for ann in anns])
        return labels
    

    def get_bboxes(self, img_id: int, cat_id: list=[0], iscrowd=None):
        img_id = self.image_ids[img_id]
        annIds = self.coco.getAnnIds(imgIds=[img_id], catIds=cat_id, iscrowd=iscrowd)
        annotations = self.coco.loadAnns(annIds)

        bboxes = []
        for ann in annotations:
            if 'bbox' in ann and ann['bbox']:
                x, y, w, h = ann['bbox']
                bboxes.append([x, y, x + w, y + h])
            elif 'segmentation' in ann and ann['segmentation']:
                for polygon in ann['segmentation']:
                    xs = polygon[0::2]
                    ys = polygon[1::2]
                    x_min = min(xs)
                    y_min = min(ys)
                    x_max = max(xs)
                    y_max = max(ys)
                    bboxes.append([x_min, y_min, x_max, y_max])

        return bboxes

        # """
        # Get bounding boxes for all annotations of a given image.
        
        # Parameters:
        # - img_id: The COCO image id.
        
        # Returns:
        # - bboxes: A list of bounding boxes, each in the format [x_min, y_min, x_max, y_max].
        # """
        # img_id = self.image_ids[img_id]
        # ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=cat_id, iscrowd=iscrowd)
        # annotations = self.coco.loadAnns(ann_ids)

        # bboxes = []
        # for ann in annotations:
        #     # This assumes polygon annotations which are stored as [x1, y1, x2, y2, ..., xn, yn]
        #     if 'segmentation' in ann and ann['segmentation']:
        #         for polygon in ann['segmentation']:
        #             xs = polygon[0::2]  # Extract every other element starting from 0
        #             ys = polygon[1::2]  # Extract every other element starting from 1
        #             x_min = min(xs)
        #             y_min = min(ys)
        #             x_max = max(xs)
        #             y_max = max(ys)
        #             width = x_max - x_min
        #             height = y_max - y_min
        #             bboxes.append([x_min, y_min, x_max, y_max])
        # # bboxes = [ann['bbox'] for ann in annotations]
        # # bboxes = [[x, y, x + w, y + h] for x, y, w, h in bboxes]

        # return bboxes


    def masks_to_boxes(self, masks):
        """
        Convert masks of shape (H, W, N) to a list of bounding boxes in the format of (x_min, y_min, x_max, y_max).
        
        Args:
        - masks (numpy.ndarray): A binary mask array of shape (H, W, N) where N is the number of masks.
        
        Returns:
        - numpy.ndarray: An array of bounding boxes of shape (N, 4).
        """
        num_masks = masks.shape[2]
        boxes = np.zeros((num_masks, 4), dtype=np.float32)

        for i in range(num_masks):
            mask = masks[:, :, i]
            # Find the boundary of the mask
            pos = np.where(mask)
            if pos[0].size > 0 and pos[1].size > 0:
                x_min = np.min(pos[1])
                y_min = np.min(pos[0])
                x_max = np.max(pos[1])
                y_max = np.max(pos[0])
                boxes[i, :] = [x_min, y_min, x_max, y_max]
            else:
                # In case the mask is empty, return an invalid box
                boxes[i, :] = [0, 0, 0, 0]

        return boxes
    
    
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