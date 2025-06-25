import albumentations as A
from .transforms.copy_paste import CopyPaste
import cv2
import numpy as np
from configs import cfg

# --- augmentation functions ---
def lsj_transforms(cfg: cfg):
    """Large Scale Jittering (LSJ) augmentation pipeline."""
    size = cfg.dataset.train_dataset.size
    return A.Compose([
        A.LongestMaxSize(max_size=max(size)),
        A.PadIfNeeded(*size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomScale(scale_limit=(0.1, 1), p=1, interpolation=1),  # scale 0.1x to 2.0x
        A.PadIfNeeded(*size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomCrop(*size),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=1),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels', 'indices']))


def ssj_transforms(cfg: cfg):
    """Small Scale Jittering (SSJ) augmentation pipeline (milder scale jitter)."""
    size = cfg.dataset.train_dataset.size
    return A.Compose([
        A.LongestMaxSize(max_size=max(size)),
        A.PadIfNeeded(*size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomScale(scale_limit=(-0.2, 0.5), p=1, interpolation=1),  # scale 0.8x to 1.5x
        A.PadIfNeeded(*size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomCrop(*size),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels', 'indices']))


# --- augmentation registry ---
AUGMENTATIONS = {
    "lsj": lsj_transforms,
    "ssj": ssj_transforms,
}


# --- factory functions ---
def get_train_transforms(cfg: cfg):
    """Factory function to get training transforms based on config."""
    aug_type = getattr(cfg.dataset.train_dataset, "augmentation", "lsj")
    if aug_type not in AUGMENTATIONS:
        available = list(AUGMENTATIONS.keys())
        raise ValueError(f"Unknown augmentation type: '{aug_type}'. Available: {available}")
    return AUGMENTATIONS[aug_type](cfg)


def get_valid_transforms(cfg: cfg):
    """Minimal validation transforms (no augmentation)."""
    size = cfg.dataset.valid_dataset.size
    return A.Compose([
        A.LongestMaxSize(max_size=max(size)),
        A.PadIfNeeded(*size, border_mode=cv2.BORDER_CONSTANT, value=0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels', 'indices']))


# --- legacy functions ---
def train_transforms(cfg: cfg):
    """Legacy function - use get_train_transforms instead."""
    return get_train_transforms(cfg)

def valid_transforms(cfg: cfg):
    """Legacy function - use get_valid_transforms instead."""
    return get_valid_transforms(cfg)