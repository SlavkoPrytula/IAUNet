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

from utils.visualise import visualize, visualize_grid_v2, plot3d
from utils.normalize import normalize
from utils.augmentations import train_transforms, valid_transforms


# data = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/EVICAN2"

# annotations = join(data, "annotations")
# images = join(data, "images")

# image = cv2.imread(join(images, "EVICAN_eval2019", "18_Caki.jpg"))
# print(image.shape)

# visualize(images=image, path='./evican2_test_image.jpg', cmap='gray',)


# import json 
# from configs.datasets import (
#     EVICAN2, EVICAN2Easy, EVICAN2Medium, EVICAN2Difficult, 
#     LiveCell, LiveCell2Percent
#     )

# # file = open(LiveCell2Percent.eval_dataset.ann_file)
# # data = json.load(file)

# # print(data["images"])
# # print(data["annotations"][0])


# # img_folder = EVICAN2.eval_dataset.images
# # ann_file = EVICAN2.eval_dataset.ann_file

# img_folder = EVICAN2.valid_dataset.images
# ann_file = EVICAN2.valid_dataset.ann_file

# # img_folder = EVICAN2.train_dataset.images
# # ann_file = EVICAN2.train_dataset.ann_file

# # img_folder = LiveCell2Percent.valid_dataset.images
# # ann_file = LiveCell2Percent.valid_dataset.ann_file

# coco = COCO(ann_file)
# print(coco.info())


# def get_mask(coco, img_id: int, cat_id: list = [0], iscrowd=None):
#     img_info = coco.loadImgs([img_id])[0] 
#     annIds = coco.getAnnIds(imgIds=img_id, catIds=cat_id, iscrowd=iscrowd)
#     anns = coco.loadAnns(annIds)
    
#     h, w = img_info['height'], img_info['width']
#     mask = np.zeros((len(anns), h, w))
#     for i, ann in enumerate(anns):
#         _mask = coco.annToMask(ann)
#         if _mask.shape != (h, w):
#             _mask = cv2.resize(_mask, (w, h), interpolation=cv2.INTER_NEAREST)
#         mask[i] = _mask
    
#     mask = np.transpose(mask, (1, 2, 0))
    
#     if len(anns) == 1:
#         mask = mask.squeeze(-1)

#     return mask


# def get_image(coco, img_id, img_folder):
#     img_info = coco.loadImgs(img_id)[0]
#     img_path = join(img_folder, img_info['file_name'])
#     image = cv2.imread(img_path, -1)

#     if image is None:
#         raise FileNotFoundError(f"Image with id {img_id} not found in path: {img_path}")

#     return image



# # img_id = 10
# image_ids = coco.getImgIds()
# # print(image_ids)
# # raise

# img_id = image_ids[0]  

# image = get_image(coco, img_id, img_folder=img_folder)
# mask = get_mask(coco, img_id, cat_id=[1], iscrowd=0)

# print(image.shape)
# print(mask.shape)

# mask = flatten_mask(mask, axis=-1)

# visualize(images=image, path='./evican2_test_image.jpg')
# visualize(images=mask, path='./evican2_test_mask.jpg')






# sample = "F4_02_2_171Z5_RFP_001"
# segm = "F4_02_2_17Z5_RFP_001_Simple Segmentation"

# data_dir = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/CHO-K1/U-Net3 training"
# # data_dir = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/CHO-K1/Ilastik training/Manual ground truth for Ilastik"
# image = cv2.imread(join(data_dir, "Flat-field corrected fluorescence images", f"{sample}.tif"), -1)

# mask = cv2.imread(join(data_dir, "Manual ground truth", f"{sample}.png")).astype(np.uint8)
# mask2 = cv2.imread(join("/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/CHO-K1/Ilastik training/Class balanced RF-FL-1 output with added ground truth and background class", f"{segm}.png"), -1)
# print(image.shape)
# print(mask.shape)
# print(mask2.shape)


# visualize(images=image, path='./CHO-K1_test_image.jpg')
# visualize(images=mask[..., 1], path='./CHO-K1_test_mask.jpg')
# visualize(images=mask2, path='./CHO-K1_test_mask2.jpg')




# data_dir = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/CHO-K1/Ilastik training/"
# sample = "F4_02_1_14Z0_Bright Field_001"

# image = cv2.imread(join(data_dir, "Raw bright-field and fluorescence images", f"F4_02_2_14Z3_RFP_001.tif"), -1)

# mask = cv2.imread(join(data_dir, "Manual ground truth for Ilastik", f"F4_02_2_14Z3_RFP_001.tif"))

# # mask2 = cv2.imread(join("/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/CHO-K1/Ilastik training/Class balanced RF-FL-1 output with added ground truth and background class", f"{segm}.png"), -1)
# print(image.shape)
# print(mask.shape)
# # print(mask2.shape)


# visualize(images=image, mask=mask[..., 1], path='./CHO-K1_test_image.jpg', figsize=[30, 15])
# visualize(images=mask[..., 1], path='./CHO-K1_test_mask.jpg')
# visualize(images=mask2, path='./CHO-K1_test_mask2.jpg')




# data_dir = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/YeastNet"
# sample1 = "z2_t_000_000_001_RFP.tif"
# sample2 = "z2_t_000_000_001_GFP.tif"

# # image1 = cv2.imread(join(data_dir, "Images/RFP/", f"{sample1}"), -1)
# # image2 = cv2.imread(join(data_dir, "Images/GFP/", f"{sample2}"), -1)
# image1 = cv2.imread(join(data_dir, "Images/Z1", f"im000.tif"), -1)
# image2 = cv2.imread(join(data_dir, "Images/Z2", f"im050.tif"), -1)
# image3 = cv2.imread(join(data_dir, "Images/Z3", f"im100.tif"), -1)
# # image1 = cv2.resize(image1, (1024, 1024))
# # image2 = cv2.resize(image2, (1024, 1024))

# mask = np.load(join(data_dir, "Images/Masks/", "mask000.npy"))

# print(image1.shape, image2.shape, image3.shape)
# print(mask.shape)

# visualize(image1=image1, image2=image2, image3=image3, mask=mask, path='./YeastNet_test_image.jpg', figsize=[40, 10])




data_dir = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/YeastNet/Datasets/DSDataset1"
sample = "000"

image = cv2.imread(join(data_dir, "Images/", f"im{sample}.tif"))
mask = np.load(join(data_dir, "Masks", f"mask{sample}.npy"))

print(image.shape)
print(mask.shape)

visualize(image=image, mask=mask, path='./YeastNet_test_image.jpg', figsize=[30, 15])