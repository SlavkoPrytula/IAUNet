import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import cv2
from tqdm import tqdm

import sys
sys.path.append('.')
from visualizations.coco_vis import getNPMasks, _visualize_masks



def visualize_masks(img, masks, shape, alpha=1, draw_border=False, static_color=False, 
                    path=None, show_img=False, figsize=[20, 10], dpi=100, ax=None):
    img = F.interpolate(img.unsqueeze(0).unsqueeze(0), size=shape, 
                        mode="bilinear", align_corners=False).squeeze(0).squeeze(0)

    masks = F.interpolate(masks.unsqueeze(0), size=shape, 
                          mode="bilinear", align_corners=False).squeeze(0)

    # Convert masks to a format that _visualize_masks can use
    masks = getNPMasks(masks, shape, alpha=alpha, static_color=static_color)
    
    if ax is None:
        # If no axis is provided, create a new figure and axis
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    ax.imshow(img, cmap='gray')
    _visualize_masks(ax, masks, draw_border)

    # Customize axis
    ax.axis('off')
    ax.set_xlim(0, shape[1] - 1)
    ax.set_ylim(shape[0] - 1, 0)

    # Save to path if specified
    if path and ax is None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close(fig)


def visualize(coco_json_path, images_dir, output_path, output_name, img_ids=None, alpha=0.65, num_cols=4):
    coco = COCO(coco_json_path)
    img_ids = img_ids or coco.getImgIds()
    num_imgs = len(img_ids)
    num_rows = 2

    # fig_count = int(np.ceil(num_imgs / num_cols))
    fig_count = 3

    for fig_idx in range(fig_count):
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 10), dpi=100)
        fig.subplots_adjust(wspace=0.05, hspace=0.01, left=0, right=1, bottom=0, top=1)

        for i, img_idx in tqdm(enumerate(img_ids[fig_idx * num_cols:(fig_idx + 1) * num_cols])):
            img_info = coco.loadImgs([img_idx])[0]
            img_path = os.path.join(images_dir, img_info['file_name'])

            # Read and normalize the image
            img = cv2.imread(img_path, -1) / 255.0 
            img = torch.tensor(img, dtype=torch.float32)

            if img.ndim == 3 and img.shape[2] == 3:
                img = img[..., 0]
            elif img.ndim == 3 and img.shape[2] == 4:
                img = img[..., 0]
            elif img.ndim == 2:
                pass
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")
            
            H, W = img_info['height'], img_info['width']

            # Load and prepare masks
            ann_ids = coco.getAnnIds(imgIds=[img_idx])
            anns = coco.loadAnns(ann_ids)
            masks = torch.zeros((len(anns), H, W), dtype=torch.float32)
            for j, ann in enumerate(anns):
                mask = coco.annToMask(ann)
                masks[j] = torch.tensor(mask, dtype=torch.float32)

            H, W = 512, 512
            
            # Resize images and masks to match the target shape
            # pad if needed and resize

            # pad
            max_shape = max(img.shape[0], img.shape[1])
            pad_h = (max_shape - img.shape[0]) // 2
            pad_w = (max_shape - img.shape[1]) // 2
            img = F.pad(img, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
            masks = F.pad(masks, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)

            # resize
            img_resized = F.interpolate(img.unsqueeze(0).unsqueeze(0), size=(H, W), 
                                        mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
            masks_resized = F.interpolate(masks.unsqueeze(0), size=(H, W), 
                                          mode="bilinear", align_corners=False).squeeze(0)
                                          

            # Top row: image with masks
            visualize_masks(img_resized, masks_resized, shape=(H, W), 
                            alpha=alpha, draw_border=True, static_color=False, ax=axes[0, i])

            axes[1, i].imshow(img_resized, cmap='gray')
            axes[1, i].axis('off')

        for ax in axes.flatten()[num_imgs % num_cols:]:
            ax.axis('off')

        save_img_path = os.path.join(output_path, f'{output_name}_{fig_idx + 1}.png')
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(save_img_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)


# revvity-25
# coco_json_path = '/project/project_465001327/datasets/Revvity-25/annotations/train.json'
# images_dir = '/project/project_465001327/datasets/Revvity-25/images'
# output_path = './'
# visualize(coco_json_path, images_dir, output_path, num_cols=4)


# # livecell
# coco_json_path = '/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/LiveCell/crop_512x512/coco/annotations/livecell_coco_test.json'
# images_dir = '/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/LiveCell/crop_512x512/coco/images/livecell_test_images'
# output_path = './cvpr/datasets/'
# visualize(coco_json_path, images_dir, output_path, output_name='livecell_dataset_images_set', num_cols=4)



# isbi2014
# coco_json_path = '/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/ISBI2014/coco/annotations/isbi_test.json'
# images_dir = '/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/ISBI2014/coco/isbi_test'
# output_path = './cvpr/datasets/'
# visualize(coco_json_path, images_dir, output_path, output_name='isbi2014_dataset_images_set', num_cols=4)


# evican2
# coco_json_path = '/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/EVICAN2/coco/annotations/EVICAN2/processed/instances_train2019_EVICAN2_cell.json'
# images_dir = '/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/EVICAN2/coco/images/EVICAN_train2019'
coco_json_path = '/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/EVICAN2/coco/annotations/EVICAN2/processed/instances_eval2019_easy_EVICAN2_cell.json'
images_dir = '/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/EVICAN2/coco/images/EVICAN_eval2019'
output_path = './cvpr/datasets/evican2'
visualize(coco_json_path, images_dir, output_path, output_name='evican2_easy_dataset_images_set', num_cols=4)


coco_json_path = '/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/EVICAN2/coco/annotations/EVICAN2/processed/instances_eval2019_medium_EVICAN2_cell.json'
images_dir = '/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/EVICAN2/coco/images/EVICAN_eval2019'
output_path = './cvpr/datasets/evican2'
visualize(coco_json_path, images_dir, output_path, output_name='evican2_medium_dataset_images_set', num_cols=4)


coco_json_path = '/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/EVICAN2/coco/annotations/EVICAN2/processed/instances_eval2019_difficult_EVICAN2_cell.json'
images_dir = '/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/EVICAN2/coco/images/EVICAN_eval2019'
output_path = './cvpr/datasets/evican2'
visualize(coco_json_path, images_dir, output_path, output_name='evican2_difficult_dataset_images_set', num_cols=4)
