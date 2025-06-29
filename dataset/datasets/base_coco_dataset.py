import cv2
import numpy as np
from os.path import join
import os.path as osp
import sys
import warnings

sys.path.append('.')

import torch
import torch.nn.functional as F
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from configs import cfg

from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Union


@dataclass
class DataSample:
    image: torch.Tensor
    masks: Optional[torch.Tensor] = None
    bboxes: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    metadata: Optional[dict] = None


class COCOAnnotationParser:
    """
    COCO annotation parser with robust error handling and configurability.
    
    Responsibilities:
    - Parse COCO annotations into standardized format
    - Handle edge cases (empty annotations, invalid bboxes, etc.)
    - Provide configurable output formats
    - Validate annotation integrity
    """
    
    def __init__(self, 
                 coco: COCO,
                 bbox_format: str = 'xyxy',
                 filter_empty: bool = True,
                 min_bbox_size: float = 1.0,
                 use_crowd: bool = False):
        """
        Args:
            coco: COCO API instance
            bbox_format: 'xyxy' or 'xywh' for bbox output format
            filter_empty: Whether to filter out empty/invalid annotations
            min_bbox_size: Minimum bbox size (width or height) to be considered valid
            use_crowd: Whether to include crowd annotations
        """
        if bbox_format not in ['xyxy', 'xywh']:
            raise ValueError(f"bbox_format must be 'xyxy' or 'xywh', got {bbox_format}")
        if min_bbox_size < 0:
            raise ValueError(f"min_bbox_size must be >= 0, got {min_bbox_size}")
            
        self.coco = coco
        self.bbox_format = bbox_format
        self.filter_empty = filter_empty
        self.min_bbox_size = min_bbox_size
        self.use_crowd = use_crowd
        
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {cat_id: idx for idx, cat_id in enumerate(self.cat_ids)}
        
    def parse_annotations(self, 
                         anns: List[Dict], 
                         img_info: Dict[str, Any],
                         return_masks: bool = True,
                         return_bboxes: bool = True,
                         return_labels: bool = True) -> Dict[str, np.ndarray]:
        """
        Parse COCO annotations for a single image.
        
        Args:
            anns: List of COCO annotation dictionaries
            img_info: Image info dictionary with 'height', 'width', etc.
            return_masks: Whether to include instance masks
            return_bboxes: Whether to include bounding boxes
            return_labels: Whether to include class labels
            
        Returns:
            Dictionary containing parsed annotations with keys:
            - 'masks': (H, W, N) format
            - 'bboxes': (N, 4) in specified format
            - 'labels': (N,) class indices
        """
        if not anns:
            return self._get_empty_annotations(img_info, return_masks, return_bboxes, return_labels)
        
        valid_anns = self._filter_annotations(anns, img_info) if self.filter_empty else anns
        
        if not valid_anns:
            return self._get_empty_annotations(img_info, return_masks, return_bboxes, return_labels)
        
        results = {}
        if return_bboxes:
            results['bboxes'] = self._extract_bboxes(valid_anns, img_info)
        if return_masks:
            results['masks'] = self._extract_masks(valid_anns, img_info)
        if return_labels:
            results['labels'] = self._extract_labels(valid_anns)
            
        return results
    
    def _filter_annotations(self, anns: List[Dict], img_info: Dict[str, Any]) -> List[Dict]:
        """Filter out invalid annotations."""
        valid_anns = []
        img_h, img_w = img_info['height'], img_info['width']
        
        for ann in anns:
            if not self.use_crowd and ann.get('iscrowd', 0):
                continue
            if ann.get('ignore', False):
                continue
                
            if 'bbox' in ann and ann['bbox']:
                x, y, w, h = ann['bbox']

                if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                    warnings.warn(f"Bbox {ann['bbox']} is outside image bounds {img_w}x{img_h}")
                    continue
                if w < self.min_bbox_size or h < self.min_bbox_size:
                    continue
                if ann.get('area', 0) <= 0:
                    continue
                    
            elif 'segmentation' in ann and ann['segmentation']:
                if not self._is_valid_segmentation(ann['segmentation']):
                    continue
            else:
                continue
            valid_anns.append(ann)
            
        return valid_anns
    
    def _is_valid_segmentation(self, segmentation: Union[List, Dict]) -> bool:
        """Check if segmentation is valid."""
        if isinstance(segmentation, dict):
            return True
        elif isinstance(segmentation, list):
            for poly in segmentation:
                if len(poly) < 6:
                    return False
            return True
        return False
    
    def _extract_bboxes(self, anns: List[Dict], img_info: Dict[str, Any]) -> np.ndarray:
        """Extract bounding boxes in specified format."""
        bboxes = []
        for ann in anns:
            if 'bbox' in ann and ann['bbox']:
                x, y, w, h = ann['bbox']
                if self.bbox_format == 'xyxy':
                    bbox = [x, y, x + w, y + h]
                else:
                    bbox = [x, y, w, h]
                bboxes.append(bbox)
            elif 'segmentation' in ann and ann['segmentation']:
                bbox = self._bbox_from_segmentation(ann['segmentation'], img_info)
                bboxes.append(bbox)
            else:
                if self.bbox_format == 'xyxy':
                    bboxes.append([0, 0, 0, 0])
                else:
                    bboxes.append([0, 0, 0, 0])

        bboxes = np.array(bboxes, dtype=np.float32)
        return bboxes
    
    def _bbox_from_segmentation(self, segmentation: Union[List, Dict], img_info: Dict[str, Any]) -> List[float]:
        """Compute bbox from segmentation."""
        if isinstance(segmentation, dict):
            try:
                mask = self.coco.annToMask({'segmentation': segmentation, 'bbox': [0, 0, img_info['width'], img_info['height']]})
                y_indices, x_indices = np.where(mask > 0)
                if len(x_indices) == 0:
                    return [0, 0, 0, 0] if self.bbox_format == 'xywh' else [0, 0, 0, 0]
                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()
            except Exception as e:
                warnings.warn(f"Failed to compute bbox from RLE: {e}")
                return [0, 0, 0, 0]
        else:
            xs, ys = [], []
            for poly in segmentation:
                xs.extend(poly[0::2])
                ys.extend(poly[1::2])
            if not xs or not ys:
                return [0, 0, 0, 0] if self.bbox_format == 'xywh' else [0, 0, 0, 0]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
        if self.bbox_format == 'xyxy':
            return [x_min, y_min, x_max, y_max]
        else:  # xywh
            return [x_min, y_min, x_max - x_min, y_max - y_min]
    
    def _extract_masks(self, anns: List[Dict], img_info: Dict[str, Any]) -> np.ndarray:
        """Extract instance masks."""
        h, w = img_info['height'], img_info['width']
        masks = np.zeros((h, w, len(anns)), dtype=np.uint8)
        
        for i, ann in enumerate(anns):
            try:
                mask = self.coco.annToMask(ann)
                masks[:, :, i] = mask
            except Exception as e:
                warnings.warn(f"Failed to convert annotation to mask: {e}")
                continue
        return masks
    
    def _extract_labels(self, anns: List[Dict]) -> np.ndarray:
        """Extract class labels."""
        labels = np.array([self.cat2label[ann['category_id']] for ann in anns], dtype=np.int64)
        return labels
    
    def _get_empty_annotations(self, 
                              img_info: Dict[str, Any],
                              return_masks: bool,
                              return_bboxes: bool, 
                              return_labels: bool) -> Dict[str, np.ndarray]:
        """Return empty annotations for images with no valid annotations."""
        results = {}
        if return_bboxes:
            results['bboxes'] = np.zeros((0, 4), dtype=np.float32)
        if return_masks:
            h, w = img_info['height'], img_info['width']
            results['masks'] = np.zeros((h, w, 0), dtype=np.uint8)
        if return_labels:
            results['labels'] = np.zeros((0,), dtype=np.int64)
        return results
    

class BaseCOCODataset(Dataset):
    def __init__(self, cfg: cfg, dataset_type="train", transform=None, 
                 return_masks=True, return_bboxes=True, return_labels=True,
                 bbox_format='xyxy', filter_empty=True, min_bbox_size=1.0, 
                 use_crowd=False, size_divisibility=32, **kwargs):
        """
        Flexible COCO dataset with configurable parameters.
        
        Args:
            cfg: Configuration object
            dataset_type: Split type ('train', 'valid', 'eval')
            transform: Augmentation transforms
            return_masks: Whether to return instance masks
            return_bboxes: Whether to return bounding boxes
            return_labels: Whether to return class labels
            filter_empty: Whether to filter empty annotations
            min_bbox_size: Minimum bbox size for filtering
            use_crowd: Whether to include crowd annotations
            size_divisibility: Size divisibility for image dimensions
            **kwargs: Additional parameters for future extensibility
        """
        if dataset_type == "train":
            self.img_folder = cfg.dataset.train_dataset.images
            self.ann_file = cfg.dataset.train_dataset.ann_file
        elif dataset_type == "valid":
            self.img_folder = cfg.dataset.valid_dataset.images
            self.ann_file = cfg.dataset.valid_dataset.ann_file
        elif dataset_type == "test":
            self.img_folder = cfg.dataset.test_dataset.images
            self.ann_file = cfg.dataset.test_dataset.ann_file
        elif dataset_type == "occ":
            raise NotImplementedError

        self.cfg = cfg
        self.coco = COCO(join(self.ann_file))
        
        self.parser = COCOAnnotationParser(
            self.coco,
            bbox_format=bbox_format,
            filter_empty=filter_empty,
            min_bbox_size=min_bbox_size,
            use_crowd=use_crowd
        )
        
        self.image_ids = self.coco.getImgIds()
        self.transform = transform
        self.return_masks = return_masks
        self.return_bboxes = return_bboxes
        self.return_labels = return_labels
        self.size_divisibility = size_divisibility
        self.extra_config = kwargs
        
        self.mean = torch.Tensor(cfg.dataset.mean).view(-1, 1, 1)
        self.std = torch.Tensor(cfg.dataset.std).view(-1, 1, 1)
        self.total_size = len(self.image_ids)

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        annIds = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(annIds)

        image = self.get_image(img_info)

        parsed = self.parser.parse_annotations(
            anns, img_info,
            return_masks=self.return_masks,
            return_bboxes=self.return_bboxes,
            return_labels=self.return_labels
        )
        masks = parsed.get('masks')
        bboxes = parsed.get('bboxes')
        labels = parsed.get('labels')
        # TODO: add padding_mask

        assert not (self.return_masks == False and self.return_bboxes == False), \
            "At least one of return_masks or return_bboxes must be True."
        keep = np.arange(bboxes.shape[0] if bboxes is not None else masks.shape[-1])

        if self.transform:
            transform_kwargs = {'image': image, 'indices': keep}
            if self.return_masks and masks is not None:
                transform_kwargs['mask'] = masks
            if self.return_bboxes and bboxes is not None:
                transform_kwargs['bboxes'] = bboxes
            if self.return_labels and labels is not None:
                transform_kwargs['labels'] = labels

            data = self.transform(**transform_kwargs)
            image = data['image']
            keep = data['indices']
            masks = data.get('mask', None)
            bboxes = data.get('bboxes', None)
            labels = data.get('labels', None)

            if self.return_bboxes == False:
                masks, keep = self.filter_empty_masks(masks, return_idx=True)

        image = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32)
        if masks is not None:
            masks = torch.tensor(masks.transpose((2, 0, 1)), dtype=torch.float32)
            # if we return bboxes, we have the keep indices
            if self.return_bboxes != False:
                masks = masks[keep]

        if bboxes is not None:
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            h, w = image.shape[-2:]
            if bboxes.numel() > 0:
                if self.parser.bbox_format == 'xyxy':
                    bboxes = box_xyxy_to_cxcywh(bboxes) / torch.tensor([w, h, w, h], dtype=torch.float32)
                else:
                    bboxes = bboxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            else:
                bboxes = torch.zeros((0, 4), dtype=torch.float32)

        if labels is not None:
            labels = torch.tensor(labels, dtype=torch.int64)
            # if we don't return bboxes, we still need to filter labels
            if self.return_bboxes == False:
                labels = labels[keep]

        resized_shape = image.shape[-2:]

        # normalize image.
        image = (image - self.mean) / self.std

        # pad images and segmentations here.
        if self.size_divisibility > 1:
            image_size = (image.shape[-2], image.shape[-1])
            pad_h = (self.size_divisibility - image_size[0] % self.size_divisibility) % self.size_divisibility
            pad_w = (self.size_divisibility - image_size[1] % self.size_divisibility) % self.size_divisibility
            padding_size = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
            # pad image
            image = F.pad(image, padding_size, value=0)
            # pad masks
            if masks is not None:
                masks = F.pad(masks, padding_size, value=0)

        datasample = {
            "image": image, # (H, W)
            "instance_masks": masks, # (N, H, W)
            "labels": labels, # (N)
            "bboxes": bboxes, # (N, 4)
            "resized_shape": resized_shape, # (res_h, res_w)
        }
        metadata = self.img_infos(img_info, img_id)
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
        
