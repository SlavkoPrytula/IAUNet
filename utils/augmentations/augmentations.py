import albumentations as A
from .transforms.copy_paste import CopyPaste
import cv2
from configs import cfg


def train_transforms(cfg: cfg):
    _transforms = A.Compose([
        A.Resize(*cfg.train.size),
        A.RandomScale(scale_limit=(-0.2, 0.5), p=1, interpolation=1),
        A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT),
        A.RandomCrop(512, 512),

        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=1),

    ], 
    # bbox_params=A.BboxParams(format='pascal_voc')
    )
    return _transforms


def valid_transforms(cfg: cfg):
    _transforms = A.Compose([
        A.Resize(*cfg.valid.size),
    ], 
    # bbox_params=A.BboxParams(format='pascal_voc')
    )
    return _transforms
