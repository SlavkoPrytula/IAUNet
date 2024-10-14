import sys
sys.path.append("./")

from dataset.datasets.brightfiled import Brightfield_Dataset
from utils.augmentations import train_transforms, valid_transforms
from configs import cfg
from tqdm import tqdm
import numpy as np
import os
from os.path import join
import cv2


# dataset = Brightfield_Dataset(cfg, dataset_type="train",
#                               normalization=False,
#                               transform=False)

# means = []
# stds = []
# for data in tqdm(dataset):
#     image = data['image'].float().numpy()
#     image = np.transpose(image, (1, 2, 0))

#     mean_per_channel = np.mean(image.reshape(-1, 3), axis=0)
#     std_per_channel = np.std(image.reshape(-1, 3), axis=0)

#     means.append(mean_per_channel)
#     stds.append(std_per_channel)
#     # print("="*10)

# # print(f"Means: {means}")
# # print(f"Stds: {stds}")
# # print()

# mean = np.mean(means, axis=0)
# std = np.mean(stds, axis=0)
# print(f"Mean: {mean}")
# print(f"Std: {std}")


# cv2.setUseOpenVX(False)

# data_root = '/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/NeurlPS22-CellSeg/coco/images_orig'
# save_root = '/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/NeurlPS22-CellSeg/coco/images2'
# dataset = os.listdir(data_root)

# os.makedirs(save_root, exist_ok=True)

# # Define a maximum allowed size, for example, 5000x5000 pixels
# max_size = (5000, 5000)

# for data in tqdm(dataset):
#     image_path = os.path.join(data_root, data)
#     image_path_save = os.path.join(save_root, data)

#     # Read the image
#     image = cv2.imread(image_path, -1).astype(np.float32)

    
#     # Normalize the image to 0-255 range (optional)
#     image /= image.max()
#     image *= 255.
#     image = image.astype(np.uint8)

#     # Save the resized image
#     cv2.imwrite(image_path_save, image)

# print("Images have been resized if necessary and saved.")


import matplotlib.pyplot as plt

data_root = '/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/NeurlPS22-CellSeg/coco_new/images'
dataset = os.listdir(data_root)[100]

means = []
stds = []

for i, data in enumerate(tqdm(dataset)):
    image_path = os.path.join(data_root, data)
    image = cv2.imread(image_path, -1)

    if image is None:
        print(f"Image {data} could not be loaded.")
        continue

    image = image.astype(np.float32)

    if len(image.shape) == 2 or image.shape[-1] == 1:
        image = np.stack((image,) * 3, axis=-1)

    mean_per_image = np.mean(image.reshape(-1, 3), axis=0)
    std_per_image = np.std(image.reshape(-1, 3), axis=0)
    means.append(mean_per_image)
    stds.append(std_per_image)

mean_dataset = np.mean(means, axis=0)
std_dataset = np.mean(stds, axis=0)

print(f"Dataset Mean: {mean_dataset}")
print(f"Dataset Std: {std_dataset}")