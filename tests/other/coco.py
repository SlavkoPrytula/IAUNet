import sys
sys.path.append("./")

import numpy as np
import cv2
import json

# from utils.visualise import visualize, visualize_grid_v2, plot3d
# from utils.normalize import normalize
# from utils.augmentations import train_transforms, valid_transforms
# from utils.coco.mask2coco import masks2coco, get_coco_template, create_image_info, create_annotation_info

# from dataset.datasets import Brightfield_Dataset
# from configs import cfg




# train_dataset = Brightfield_Dataset(cfg, dataset_type="train",
#                             # normalization=normalize,
#                             transform=valid_transforms(cfg)
#                             )

# valid_dataset = Brightfield_Dataset(cfg, dataset_type="valid",
#                             # normalization=normalize,
#                             transform=valid_transforms(cfg)
#                             )


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



# ==================
# test/cyto_coco_512
# - brightfield_v1 dataset
# - 5 fold split
# - fold_0 val
    

# ==================
from utils.visualise import visualize
img = cv2.imread("tests/other/cyto_coco_512/images/train/image1.jpg", -1)
print(np.all(img[..., 1] == img[..., 2]))
# visualize(ch0=img[..., 0], ch1=img[..., 1], ch2=img[..., 2], path="./test_image.jpg")