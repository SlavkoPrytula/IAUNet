import numpy as np
import os.path as osp
import torch

from scipy.ndimage import binary_dilation

import sys
sys.path.append('.')

from utils.utils import flatten_mask
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

from configs import cfg

from dataset.datasets.base_coco_dataset import BaseCOCODataset
from utils.registry import DATASETS


@DATASETS.register(name="rectangle")
class Rectangle(BaseCOCODataset):
    def __init__(self, cfg: cfg, dataset_type="train", normalization=None, transform=None):
        super().__init__(cfg, dataset_type, normalization, transform)

    def get_image(self, img_id):
        image = self.get_mask(img_id, cat_id=[1], iscrowd=0)
        image = flatten_mask(image, -1)[..., 0]
        image = np.stack((image,) * 3, axis=-1)
        image = image.astype(np.float32)
        return image

    def img_infos(self, img_id):
        img_id = self.image_ids[img_id]
        metadata = {}

        img_info = self.coco.loadImgs([img_id])[0]
        img_path = img_info['file_name']
        fname, name = osp.splitext(osp.basename(img_path))

        metadata["img_id"] = img_id
        metadata["img_path"] = img_path
        metadata["ori_shape"] = [img_info["height"], img_info["width"]]
        metadata["file_name"] = fname
        metadata["coco_id"] = img_id

        return metadata

    # def __len__(self):
    #     return 20


    def __getitem__(self, idx):
        image = self.get_image(idx)
        mask = self.get_mask(idx)
        # bboxes = self.get_bboxes(idx)
        labels = self.get_labels(idx)

        if idx not in self.means:
            self.means[idx] = np.mean(image, axis=None, keepdims=True)
            self.stds[idx] = np.std(image, axis=None, keepdims=True)

        # if self.normalization:
        mean = self.means[idx]
        std = self.stds[idx]
        image = (image - mean) / std

        assert image.shape[-1] != 0
        assert mask.shape[-1] != 0

        if self.transform:
            data = self.transform(
                image=image, 
                mask=mask, 
                # bboxes=bboxes,
                # labels=labels
                )
            image = data['image']
            mask = data['mask']
            # bboxes = data['bboxes']
            # labels = data['labels']

        # (H, W, M) -> (H, W, N)
        mask, keep = self.filter_empty_masks(mask, return_idx=True) 
        # overlaps = self.get_overlaps(mask)
        # visible_mask = self.get_visible_mask(mask)
        # occluders = self.get_occluders(mask)
        # borders_mask = self.get_borders(mask)
        # bboxes = self.masks_to_boxes(mask)

        image = image.transpose((2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32)
        
        mask = mask.transpose((2, 0, 1))
        mask = torch.tensor(mask, dtype=torch.float32)

        # overlaps = np.transpose(overlaps, (2, 0, 1))
        # overlaps = torch.tensor(overlaps, dtype=torch.float32)
        # visible_mask = np.transpose(visible_mask, (2, 0, 1))
        # visible_mask = torch.tensor(visible_mask, dtype=torch.float32)

        # occluders = np.transpose(occluders, (2, 0, 1))
        # occluders = torch.tensor(occluders, dtype=torch.float32)

        # borders_mask = np.transpose(borders_mask, (2, 0, 1))
        # borders_mask = torch.tensor(borders_mask, dtype=torch.float32)

        # labels
        labels = torch.tensor(labels, dtype=torch.int64)
        labels = labels[keep]

        # bboxes
        # h, w = image.shape[-2:]
        # bboxes = torch.tensor(bboxes, dtype=torch.float32)
        # bboxes = box_xyxy_to_cxcywh(bboxes)
        # bboxes = bboxes / torch.tensor([w, h, w, h], dtype=torch.float32)


        target = {
            "image": image,
            "instance_masks": mask,
            # "occluder_masks": occluders,
            # "overlap_masks": overlaps,
            # "visible_masks": visible_mask,
            "labels": labels,
            # "borders_masks": borders_mask,
            # "occluders_bounds": occluder_bound
            # "bboxes": bboxes,
        }
        metadata = self.img_infos(idx)
        target.update(metadata)

        return target


    def get_overlaps(self, masks):
        _, _, N = masks.shape
        overlaps = np.zeros_like(masks, dtype=bool)
        all_masks_summed = np.sum(masks, axis=-1)
        for i in range(N):
            overlaps[:, :, i] = masks[:, :, i] & (all_masks_summed - masks[:, :, i] > 0)

        return overlaps
    

    def get_visible_mask(self, masks):
        _, _, N = masks.shape
        visible = np.zeros_like(masks, dtype=bool)
        all_masks_summed = np.sum(masks, axis=-1)
        for i in range(N):
            visible[:, :, i] = masks[:, :, i] - (masks[:, :, i] & (all_masks_summed - masks[:, :, i] > 0))

        return visible


    def get_occluders(self, masks):
        _, _, N = masks.shape
        occluders = np.zeros_like(masks, dtype=bool)
        overlap_matrix = np.logical_and(masks[:, :, None], masks[:, :, :, None])
        overlap_any = np.any(overlap_matrix, axis=(0, 1))
        np.fill_diagonal(overlap_any, False)

        for i in range(N):
            occluders[:, :, i] = np.any(masks[:, :, overlap_any[i, :]], axis=-1)

        return occluders


    def get_borders(self, masks, width=16):
        H, W, N = masks.shape
        border_masks = np.zeros((H, W, N), dtype=np.uint8)

        for i in range(N):
            mask = masks[:, :, i]
            # Perform binary dilation to get border regions
            border_mask = binary_dilation(mask, structure=np.ones((width, width)))
            # Exclude the original mask region
            border_mask = np.logical_and(border_mask, np.logical_not(mask))
            border_masks[:, :, i] = border_mask

        return border_masks



if __name__ == "__main__":
    from utils.visualise import visualize, visualize_grid_v2, plot3d
    from utils.augmentations import normalize
    from utils.augmentations import train_transforms, valid_transforms
    from utils.registry import DATASETS_CFG
    from visualizations import visualize_masks
    import time

    time_s = time.time()
    
    cfg.dataset = DATASETS_CFG.get("worms")

    dataset = Rectangle(cfg, dataset_type="valid", 
                      normalization=normalize,
                      transform=valid_transforms(cfg)
                      )
    
    print(len(dataset))
    
    targets = dataset[4]
    time_e = time.time()
    print(f'loaded in {time_e - time_s}(s)')

    print(targets["image"].shape, targets["ori_shape"])
    print(targets["labels"])
    # print(targets["bboxes"])

    print(targets["image"].shape, targets["instance_masks"].shape)
    print(f'std: {targets["image"].std()}, mean: {targets["image"].mean()}')

    
    # visualize(images=targets["image"][0, ...], path='./test_image.jpg', cmap='gray',)
    # visualize(images=flatten_mask(targets["masks"].numpy(), axis=0)[0], path='./test_mask.jpg', cmap='gray',)

    H, W = targets["ori_shape"]
    visualize_masks(
        img=targets["image"][0, ...],
        masks=targets["masks"],
        shape=[H, W],
        alpha=0.65,
        draw_border=True, 
        static_color=False,
        path='./test_mask.jpg'
    )
    
    
    # bbox.
    # h, w = targets["image"].shape[-2:]
    # bboxes = box_cxcywh_to_xyxy(targets["bboxes"])
    # bboxes = bboxes * torch.tensor([w, h, w, h], dtype=torch.float32)

    visualize_grid_v2(
        masks=targets["instance_masks"].numpy(), 
        # bboxes=bboxes.numpy(), 
        path='./test_inst.jpg',
        ncols=5
    )

    # visualize_grid_v2(
    #     masks=targets["occluder_masks"].numpy(), 
    #     path='./test_occl.jpg',
    #     ncols=5
    # )

    visualize_grid_v2(
        masks=targets["overlap_masks"].numpy(), 
        path='./test_overlap_mask.jpg',
        ncols=5
    )

    visualize_grid_v2(
        masks=targets["visible_masks"].numpy(), 
        path='./test_visible_mask.jpg',
        ncols=5
    )
    