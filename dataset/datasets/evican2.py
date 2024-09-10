import sys
sys.path.append('.')

import hydra
from configs import cfg
import torch

from dataset.datasets.base_coco_dataset import BaseCOCODataset
from utils.registry import DATASETS
import numpy as np
from tqdm import tqdm


@DATASETS.register(name="EVICAN2")
class EVICAN2(BaseCOCODataset):
    def __init__(self, cfg: cfg, dataset_type="train", normalization=None, transform=None):
        super().__init__(cfg, dataset_type, normalization, transform)
        self.filter_bg()
        self.total_size = len(self.image_ids)

    def filter_bg(self):
        self.image_ids = [
            img_id for img_id in self.image_ids 
            if "Background" not in self.coco.loadImgs([img_id])[0]['file_name']
        ]

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
        # labels = labels[keep]

        image = image.transpose((2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        
        masks = masks.transpose((2, 0, 1))
        masks = torch.tensor(masks, dtype=torch.float32)

        # labels
        labels = torch.tensor(labels, dtype=torch.int64)
        labels = labels[keep]

        # h, w = image.shape[-2:]
        # bboxes = torch.tensor(bboxes, dtype=torch.float32)
        # bboxes = box_xyxy_to_cxcywh(bboxes)
        # bboxes = bboxes / torch.tensor([w, h, w, h], dtype=torch.float32)

        target = {
            "image": image,
            "instance_masks": masks,
            "labels": labels,
            # "bboxes": bboxes,
        }
        metadata = self.img_infos(idx)
        target.update(metadata)

        return target


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: cfg):
    from utils.visualise import visualize, visualize_grid_v2
    from visualizations import visualize_masks
    from utils.utils import flatten_mask
    from utils.augmentations import normalize
    from utils.augmentations import train_transforms, valid_transforms
    import time
    import numpy as np
    from tqdm import tqdm

    time_s = time.time()

    dataset = EVICAN2(cfg, dataset_type="train", 
                    #   normalization=normalize,
                      transform=train_transforms(cfg)
                      )
    
    print(len(dataset))

    targets = dataset[10]
    time_e = time.time()
    print(f'loaded in {time_e - time_s}(s)')

    print(targets["image"].shape, targets["ori_shape"])
    print(targets["labels"])
    print(len(targets["labels"]), len(targets["instance_masks"]))

    print(targets["image"].shape, targets["instance_masks"].shape)
    print(f'std: {targets["image"].std()}, mean: {targets["image"].mean()}')

    visualize(
        images=targets["image"][0, ...], 
        path='./test_image.jpg', 
        cmap='gray'
    )

    H, W = targets["ori_shape"]
    visualize_masks(
        img=targets["image"][0, ...],
        masks=targets["instance_masks"],
        shape=[H, W],
        alpha=0.65,
        draw_border=True, 
        static_color=False,
        path='./test_mask.jpg',
        show_img=True
    )

    visualize_grid_v2(
        masks=targets["instance_masks"].numpy(), 
        path='./test_inst.jpg',
        ncols=5
    )

if __name__ == "__main__":
    main()




# import cv2
# import numpy as np
# import pandas as pd
# import json
# from os.path import join
# import re

# import cv2
# import scipy.ndimage as ndi
# from scipy.ndimage import binary_dilation

# import sys
# sys.path.append('.')

# import torch
# from torch.utils.data import Dataset

# from utils.utils import flatten_mask
# from pycocotools.coco import COCO
# from configs import cfg

# from dataset.prepare_dataset import get_folds
# from utils.registry import DATASETS



# images = np.stack([self.get_image(idx) for idx in range(len(self.image_ids))], axis=0)
# mean = np.mean(images, axis=(0, 1, 2))
# std = np.std(images, axis=(0, 1, 2))
# print(mean)
# print(std)
# raise




# @DATASETS.register(name="EVICAN2")
# class EVICAN2(Dataset):
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
        

#         self.coco = COCO(join(self.ann_file))
#         self.image_ids = self.coco.getImgIds()
#         if dataset_type == "train":
#             np.random.shuffle(self.image_ids)
        
#         self.cfg = cfg
#         self.dataset_type = dataset_type
#         self.normalization = normalization
#         self.transform = transform

#         # self.total_size = 30 if dataset_type == "train" else 10
#         self.total_size = len(self.image_ids)

#     def __len__(self):
#         return self.total_size
#         # return len(self.coco.imgs)

#     def __getitem__(self, idx):
#         idx = self.image_ids[idx]
#         image = self.get_image(idx, img_folder=self.img_folder)
#         mask = self.get_mask(idx, cat_id=[1], iscrowd=0)
#         # labels = self.get_labels(idx, cat_id=[1], iscrowd=0)

#         if self.normalization:
#             image = self.normalization(image)

#         if self.transform:
#             data = self.transform(
#                 image=image, 
#                 mask=mask, 
#                 )
#             image = data['image']
#             mask = data['mask']

#         # (H, W, M) -> (H, W, N)
#         mask, keep = self.filter_empty_masks(mask, return_idx=True)   

#         image = np.transpose(image, (2, 0, 1))
#         image = torch.tensor(image, dtype=torch.float32)
        
#         mask = np.transpose(mask, (2, 0, 1))
#         mask = torch.tensor(mask, dtype=torch.float32)

#         # labels = torch.tensor(labels, dtype=torch.int64)

#         # labels
#         N, _, _ = mask.shape
#         labels = torch.zeros(N, dtype=torch.int64)

#         target = {
#             "image": image,
#             "masks": mask,
#             "labels": labels,
#         }

#         return target
    

#     def get_mask(self, img_id: int, cat_id: list=[0], iscrowd=None):
#         img_info = self.coco.loadImgs([img_id])[0] 
#         annIds = self.coco.getAnnIds(imgIds=img_id, catIds=cat_id, iscrowd=iscrowd)
#         anns = self.coco.loadAnns(annIds)
        
#         h, w = img_info['height'], img_info['width']

#         if "Background" in img_info['file_name']:
#             mask = np.zeros((1, h, w))
#             return mask
        
#         mask = np.zeros((len(anns), h, w))
#         for i, ann in enumerate(anns):
#             _mask = self.coco.annToMask(ann)
#             if _mask.shape != (h, w):
#                 _mask = cv2.resize(_mask, (w, h), interpolation=cv2.INTER_NEAREST)
#             mask[i] = _mask
        
#         mask = np.transpose(mask, (1, 2, 0))

#         return mask


#     def get_image(self, img_id, img_folder):
#         img_info = self.coco.loadImgs(img_id)[0]
#         img_path = join(img_folder, img_info['file_name'])
#         image = cv2.imread(img_path, -1).astype(np.float32)

#         if image is None:
#             raise FileNotFoundError(f"Image with id {img_id} not found in path: {img_path}")

#         return image
    

#     def img_infos(self, img_id):
#         img_id = self.image_ids[img_id]
#         metadata = {}

#         img_info = self.coco.loadImgs(img_id)[0]
#         img_path = join(self.img_folder, img_info['file_name'])

#         metadata["img_path"] = img_path
#         metadata["height"] = img_info["height"]
#         metadata["width"] = img_info["width"]

#         return metadata
    

#     def get_labels(self, img_id: int, cat_id: list=[0], iscrowd=None):
#         annIds = self.coco.getAnnIds(imgIds=img_id, catIds=cat_id, iscrowd=iscrowd)
#         anns = self.coco.loadAnns(annIds)

#         labels = [ann['category_id'] for ann in anns]
#         return labels


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
    
#     # @staticmethod
#     # def filter_empty_masks(masks, return_idx=False):
#     #     # This presumes masks are of the shape (H, W, N) where N is the last dimension
#     #     is_empty = np.all(masks == 0, axis=(0, 1))  # Check for empty masks across the first two dimensions
#     #     kept_indices = np.where(~is_empty)[0]  # Indices of non-empty masks

#     #     # Filter out empty masks
#     #     masks = masks[:, :, kept_indices] if kept_indices.size > 0 else np.zeros((masks.shape[0], masks.shape[1], 1), dtype=masks.dtype)

#     #     if return_idx:
#     #         return masks, kept_indices
#     #     else:
#     #         return masks


        

# if __name__ == "__main__":
#     from utils.visualise import visualize, visualize_grid_v2, plot3d
#     from utils.normalize import normalize
#     from utils.augmentations import train_transforms, valid_transforms
#     import time

#     time_s = time.time()
#     dataset = EVICAN2(cfg, dataset_type="train",
#                       normalization=normalize,
#                       transform=train_transforms(cfg)
#                       )
#     targets = dataset[7]
#     time_e = time.time()
#     print(f'loaded in {time_e - time_s}(s)')

#     print(len(dataset))
#     print(targets["image"].shape)
#     # print(targets["labels"])

#     print(targets["image"].shape, targets["masks"].shape)

    
#     visualize(images=targets["image"][0, ...], path='./test_image.jpg', cmap='gray',)
#     visualize(images=flatten_mask(targets["masks"].numpy(), axis=0)[0], path='./test_mask.jpg', cmap='gray',)
    