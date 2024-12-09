import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import binary_dilation, binary_erosion
from pycocotools import mask as maskUtils
from torch.nn import functional as F
import cv2
import torch

from .palette import jitter_color, random_color


def _visualize_masks(ax, masks, draw_border=False):
    for mask in masks:
        if draw_border:
            binary_mask = mask[..., 3] > 0
            # dilation = binary_dilation(binary_mask, iterations=10)
            # erosion = binary_erosion(binary_mask, iterations=2)
            dilation = binary_dilation(binary_mask, iterations=6)
            erosion = binary_erosion(binary_mask, iterations=4)
            border = dilation & ~erosion
            
            for c in range(3):
                mask[..., c][border] = 1  # Set border color to white
        
        # mask[..., 3][border] = 1
        ax.imshow(mask)


def getMasks(coco_api, anns, shape, alpha=1, static_color=False):
    """
    Generate and return colored masks for specified annotations with optional transparency.
    :param anns: Annotations to display, each with segmentation info.
    :param alpha: Opacity level for the masks.
    :return: A numpy array of masks with applied colors and alpha transparency.
    """
    if len(anns) == 0:
        return np.zeros((0, 0, 4))

    masks = []
    for ann in anns:
        if 'segmentation' not in ann:
            continue

        t = coco_api.imgs[ann['image_id']]
        img_height, img_width = t['height'], t['width']

        if isinstance(ann['segmentation'], list):
            # polygon
            rle = maskUtils.frPyObjects(ann['segmentation'], img_height, img_width)
        else:
            # rle
            rle = ann['segmentation'] if isinstance(ann['segmentation']['counts'], list) \
                else [ann['segmentation']]
        mask = maskUtils.decode(rle)

        if ann['iscrowd'] == 1:
            color_mask = np.array([2.0,166.0,101.0])/255
        if ann['iscrowd'] == 0:
            color_mask = jitter_color([1, 0, 0]) if static_color else random_color()
        img = np.ones((img_height, img_width, 3)) * color_mask

        mask = np.dstack((img, mask[:, :, 0] * alpha))
        masks.append(mask)

    return np.array(masks) if masks else np.zeros((0, 0, 4)) 


def getNPMasks(masks, shape, alpha=1, static_color=False):
    colored_masks = []
    for mask in masks:
        color_mask = jitter_color([1, 0, 0]) if static_color else random_color()

        img_height, img_width = mask.shape
        img = np.ones((img_height, img_width, 3)) * color_mask
        colored_mask = np.dstack((img, mask * alpha))
        colored_masks.append(colored_mask)

    return np.array(colored_masks)


def visualize_coco_anns(coco_api, idx, ax, shape, alpha=1, draw_border=False, static_color=False):
    if not coco_api: # no instances were detected :(
        return 
    
    annIds = coco_api.getAnnIds(imgIds=[idx])
    anns = coco_api.loadAnns(annIds)
    masks = getMasks(coco_api, anns, shape, alpha=alpha, static_color=static_color)
    _visualize_masks(ax, masks, draw_border)


def visualize_masks(img, masks, shape, alpha=1, draw_border=False, static_color=False, path=None, show_img=False, figsize=[20, 10], dpi=100):
    img = F.interpolate(img.unsqueeze(0).unsqueeze(0), size=shape, 
                        mode="bilinear", align_corners=False).squeeze(0).squeeze(0)

    masks = F.interpolate(masks.unsqueeze(0), size=shape, 
                        mode="bilinear", align_corners=False).squeeze(0)
    
    if show_img:
        fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        ax[0].imshow(img, cmap='gray')
        ax[1].imshow(img, cmap='gray')

        masks = getNPMasks(masks, shape, alpha=alpha, static_color=static_color)
        _visualize_masks(ax[1], masks, draw_border)
        
        for a in ax:
            a.axis('off')
            a.set_xlim(0, shape[1]-1)
            a.set_ylim(shape[0], 0)
        
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        ax.imshow(img, cmap='gray')

        masks = getNPMasks(masks, shape, alpha=alpha, static_color=static_color)
        _visualize_masks(ax, masks, draw_border)
        
        ax.axis('off')
        ax.set_xlim(0, shape[1]-1)
        ax.set_ylim(shape[0]-1, 0)

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        
    plt.close(fig)



def save_coco_vis(img, gt_coco, pred_coco, idx, shape, path=None, show_img=False):
    """
    Saves a side-by-side visualization of ground truth 
    and predicted COCO annotations for a given image, with an option to show the image separately.
    """
    if isinstance(img, np.ndarray):
        img = torch.tensor(img, dtype=torch.float32)

    img = F.interpolate(img.unsqueeze(0).unsqueeze(0), size=shape, 
                        mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    img = img.cpu().numpy()

    if show_img:
        fig, ax = plt.subplots(1, 3, figsize=[30, 10])
        fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        ax[0].imshow(img, cmap='gray') # type: ignore

        ax[1].imshow(img, cmap='gray')
        visualize_coco_anns(gt_coco, idx, ax[1], shape=shape, alpha=0.65, draw_border=True, static_color=False)

        ax[2].imshow(img, cmap='gray')
        visualize_coco_anns(pred_coco, idx, ax[2], shape=shape, alpha=0.65, draw_border=True, static_color=False)

    else:
        fig, ax = plt.subplots(1, 2, figsize=[20, 10])
        fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        ax[0].imshow(img, cmap='gray')
        visualize_coco_anns(gt_coco, idx, ax[0], shape=shape, alpha=0.65, draw_border=True, static_color=False)
    
        ax[1].imshow(img, cmap='gray')
        visualize_coco_anns(pred_coco, idx, ax[1], shape=shape, alpha=0.65, draw_border=True, static_color=False)

    for a in ax:
        a.axis('off')
        a.set_xlim(0, shape[1])
        a.set_ylim(shape[0]-1, 0)

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches='tight', pad_inches=0)
    
    plt.close(fig)