import albumentations as A
from .transforms.copy_paste import CopyPaste
import cv2
import numpy as np
from configs import cfg


# def train_transforms(cfg: cfg):
#     f = np.random.uniform(0.75, 1)
#     size = cfg.dataset.train_dataset.size
#     _transforms = A.Compose([
#          A.OneOf([
#             A.RandomResizedCrop(height=size[0], 
#                                 width=size[1], 
#                                 scale=(0.5, 1.0),
#                                 ratio=(0.75, 1.33),
#                                 interpolation=cv2.INTER_LINEAR, 
#                                 p=1),
#             A.Compose([
#                 A.Resize(int(size[0] * f), int(size[1] * f)),
#                 A.PadIfNeeded(min_height=size[0], 
#                               min_width=size[1], 
#                               border_mode=cv2.BORDER_CONSTANT, 
#                               value=0)
#             ])
#         ], p=1.0),
#         A.OneOf([
#             A.VerticalFlip(p=0.5),
#             A.HorizontalFlip(p=0.5),
#             A.RandomRotate90(p=1),
#         ], p=1.0),
#     ],
#     # bbox_params=A.BboxParams(format='pascal_voc')
#     )
#     return _transforms


# def valid_transforms(cfg: cfg):
#     size = cfg.dataset.valid_dataset.size
#     _transforms = A.Compose([
#         A.Resize(*size),
#     ], 
#     # bbox_params=A.BboxParams(format='pascal_voc')
#     )
#     return _transforms


def train_transforms(cfg: cfg):
    size = cfg.dataset.train_dataset.size
    _transforms = A.Compose([
        # A.Resize(*size),
        A.LongestMaxSize(max_size=max(size)),
        A.PadIfNeeded(*size, border_mode=cv2.BORDER_CONSTANT, value=0),
        
        A.RandomScale(scale_limit=(-0.2, 1.5), p=1, interpolation=1),
        A.PadIfNeeded(*size, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomCrop(*size),

        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=1),

        A.ElasticTransform(
            alpha=10,
            sigma=10,
            interpolation=1,
            border_mode=cv2.BORDER_CONSTANT,
            approximate=True,
            p=1
        ),
    ], 
    # bbox_params=A.BboxParams(format='pascal_voc')
    )
    return _transforms


def valid_transforms(cfg: cfg):
    size = cfg.dataset.valid_dataset.size
    _transforms = A.Compose([
        # A.Resize(*size),
        A.LongestMaxSize(max_size=max(size)),
        A.PadIfNeeded(*size, border_mode=cv2.BORDER_CONSTANT, value=0),
    ], 
    # bbox_params=A.BboxParams(format='pascal_voc')
    )
    return _transforms