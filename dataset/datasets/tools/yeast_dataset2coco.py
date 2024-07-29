import sys
sys.path.append("./")

import numpy as np
import cv2
import json
from os.path import join
import os
from tqdm import tqdm

from utils.coco.mask2coco import get_coco_template, create_image_info, create_annotation_info

from configs import cfg



def categorical2sparse(mask):
    h, w = mask.shape
    N = len(np.unique(mask))
    result = np.zeros((N, h, w))
    
    # print(np.unique(mask), result.shape)

    for i in range(N):
        _mask = mask.copy()
        _mask[_mask != i] = 0
        _mask[_mask == i] = 1
        _mask = cv2.resize(_mask, (h, w))
        result[i] = _mask
    result = np.transpose(result, (1, 2, 0))
    result = result[..., 1:]
    return result


def masks2coco(masks: list, scores: list = None, file_names: list = None):
    # masks shape:  [(N, H, W), ...]
    # scores shape: [(N), ...]

    coco_dict = get_coco_template()

    # Create a new image id
    image_id = 1
    annotation_id = 1

    for i, mask_set in enumerate(masks):
        # mask_set shape: (N, H, W)

        # Create a new image info
        image_info = create_image_info(
            image_id=image_id,
            image_size=mask_set[0].shape,
            file_name=file_names[i] if file_names is not None else f"image{image_id}.jpg",
            coco_url="",
            date_captured=""
        )

        coco_dict["images"].append(image_info)

        # Loop over every instance mask
        for j, mask in enumerate(mask_set):

            # Create a new annotation info
            annotation_info = create_annotation_info(
                annotation_id=annotation_id,
                image_id=image_id,
                category_info={'is_crowd': 0, 'id': 1},
                binary_mask=mask,
                image_size=mask.shape,
            )

            if annotation_info is None:
                continue

            if scores is not None:
                annotation_info['score'] = scores[i][j]
            else:  # <---
                annotation_info['score'] = 1

            # Append annotation info to COCO dictionary
            coco_dict["annotations"].append(annotation_info)

            # Increment annotation id
            annotation_id += 1

        # Increment image id
        image_id += 1

    return coco_dict




data_dir = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/YeastNet"
N = len(os.listdir(join(data_dir, "Datasets/DSDataset1/Images")))

masks = []
file_names = []
offset = 50
for i in tqdm(range(N)):

    image = cv2.imread(join(data_dir, f"Datasets/DSDataset1/Images/im{str(i).zfill(3)}.tif"), -1)
    mask = np.load(join(data_dir, f"Datasets/DSDataset1/Masks/mask{str(i).zfill(3)}.npy"))
    
    if len(np.unique(mask)) > 50:
        continue
    mask = categorical2sparse(mask)
    mask = np.transpose(mask, (2, 0, 1))
    masks.append(mask)

    file_name = f"im{str(i).zfill(3)}.tif"
    file_names.append(file_name)

    cv2.imwrite(join(data_dir, 'coco/images/train', file_name), image)

train_coco = masks2coco(masks, file_names=file_names)

with open(join(data_dir, 'coco/annotations/train.json'), 'w') as fp:
    json.dump(train_coco, fp, indent=4)

print("Done")




data_dir = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/YeastNet"
N = len(os.listdir(join(data_dir, "Datasets/DSDataset2/Images")))

masks = []
file_names = []
offset = 50
for i in tqdm(range(N)):

    image = cv2.imread(join(data_dir, f"Datasets/DSDataset2/Images/im{str(i).zfill(3)}.tif"), -1)
    mask = np.load(join(data_dir, f"Datasets/DSDataset2/Masks/mask{str(i).zfill(3)}.npy"))
    
    if len(np.unique(mask)) > 50:
        continue
    mask = categorical2sparse(mask)
    mask = np.transpose(mask, (2, 0, 1))
    masks.append(mask)

    file_name = f"im{str(i).zfill(3)}.tif"
    file_names.append(file_name)

    cv2.imwrite(join(data_dir, 'coco/images/valid', file_name), image)

valid_coco = masks2coco(masks, file_names=file_names)

with open(join(data_dir, 'coco/annotations/valid.json'), 'w') as fp:
    json.dump(valid_coco, fp, indent=4)

print("Done")




# data_dir = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/YeastNet"
# N = len(os.listdir(join(data_dir, "Images/Z1")))

# masks = []
# file_names = []
# offset = 50
# for i in range(5):

#     image1 = cv2.imread(join(data_dir, f"Images/Z1/im{str(i).zfill(3)}.tif"), -1)
#     image2 = cv2.imread(join(data_dir, f"Images/Z2/im{str(i+offset).zfill(3)}.tif"), -1)
#     image3 = cv2.imread(join(data_dir, f"Images/Z3/im{str(i+offset*2).zfill(3)}.tif"), -1)

#     image = np.stack([image1, image2, image3], -1) 
    
#     mask = np.load(join(data_dir, f"Images/Masks/mask{str(i).zfill(3)}.npy"))
#     mask = categorical2sparse(mask)
#     mask = np.transpose(mask, (2, 0, 1))
#     masks.append(mask)

#     file_name = f"im{str(i).zfill(3)}.tif"
#     file_names.append(file_name)

#     # cv2.imwrite(join(data_dir, file_name), image)

# train_coco = masks2coco(masks, file_names=file_names)
# print(train_coco)







# masks = []
# for i in range(len(train_dataset)):
#     out = train_dataset[i]
#     img = out["image"].numpy()
#     mask = out["masks"].numpy()
    
#     img /= np.max(img)
#     img *= 255.

#     img = np.transpose(img, (1, 2, 0))
#     img = np.concatenate([img, img[:, :, -1][:, :, np.newaxis]], axis=-1)

#     cv2.imwrite(f"test/cyto_coco_512/images/train/image{i+1}.jpg", img)

#     masks.append(mask)

# train_coco = masks2coco(masks)

# with open('test/cyto_coco_512/train.json', 'w') as fp:
#     json.dump(train_coco, fp, indent=4)




# masks = []

# for i in range(len(valid_dataset)):
#     out = valid_dataset[i]
#     img = out["image"].numpy()
#     mask = out["masks"].numpy()
    
#     img /= np.max(img)
#     img *= 255.

#     img = np.transpose(img, (1, 2, 0))
#     img = np.concatenate([img, img[:, :, -1][:, :, np.newaxis]], axis=-1)

#     cv2.imwrite(f"test/cyto_coco_512/images/valid/image{i+1}.jpg", img)

#     masks.append(mask)

# train_coco = masks2coco(masks)

# with open('test/cyto_coco_512/valid.json', 'w') as fp:
#     json.dump(train_coco, fp, indent=4)




