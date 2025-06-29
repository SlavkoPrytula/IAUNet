import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import measure
from pycocotools import mask as maskUtils
from torch.nn import functional as F
import torch
from utils.box_ops import box_cxcywh_to_xyxy

from .palette import jitter_color, random_color


def _visualize_masks(ax, masks, draw_border=False, border_size=5, colors=None, border_color="same"):
    """
    Visualizes masks with optional border using the same color as the mask.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw the masks on.
        masks (list or np.ndarray): List of RGBA masks.
        draw_border (bool): Whether to draw a border around the masks. Default is False.
        border_size (int): Thickness of the border if draw_border is True. Default is 5.
        colors (list, optional): List of colors for each mask. If None, uses white for all masks.
        border_color (str): Color of the border. Accepts ['same', 'white'].
    """
    if colors is None:
        colors = [[1, 1, 1]] * len(masks)

    for i, mask in enumerate(masks):
        _mask = mask.copy()
        
        ax.imshow(_mask)
        if draw_border and i < len(colors):
            binary_mask = _mask[..., 3] > 0
            contours = measure.find_contours(binary_mask, 0.5)

            if border_color == "same":
                mask_color = colors[i]
            elif border_color == "white":
                mask_color = [1, 1, 1]
            else:
                raise ValueError("border_color must be 'same' or 'white'")
            
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], color=mask_color, linewidth=border_size)


def _visualize_bboxes(ax, bboxes, bbox_linewidth=2, colors=None):
    """
    Draw bounding boxes on the given axis.
    Args:
        ax: matplotlib axis
        bboxes: list or array of bboxes in cxcywh format, shape (N, 4)
        colors: list of RGB colors for each bbox
        shape: (H, W) tuple for scaling
        bbox_linewidth: line width in points
    """
    if colors is None:
        colors = [[1, 0, 0]] * len(bboxes)

    if bboxes is not None and len(bboxes) > 0:
        for i, bbox in enumerate(bboxes):
            if isinstance(bbox, np.ndarray):
                bbox = torch.tensor(bbox)
            x_min, y_min, x_max, y_max = bbox.int().tolist()
            bbox_color = colors[i] if i < len(colors) else [1, 0, 0]
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 edgecolor=bbox_color, facecolor='none', linewidth=bbox_linewidth, zorder=2)
            ax.add_patch(rect)


def getMasks(coco_api, anns, shape, alpha=1, static_color=False):
    """
    Generate and return colored masks for specified annotations with optional transparency.
    
    Args:
        coco_api: COCO API object.
        anns: Annotations to display, each with segmentation info.
        shape: Target shape (H, W).
        alpha: Opacity level for the masks.
        static_color: Whether to use static colors.
        
    Returns:
        tuple: (masks, colors) where masks is an array of shape (N, H, W, 4) and colors is a list of RGB values.
    """
    if len(anns) == 0:
        return np.zeros((0, shape[0], shape[1], 4)), []

    colored_masks = []
    colors = []
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
            color_mask = np.array([2.0, 166.0, 101.0]) / 255
        elif ann['iscrowd'] == 0:
            color_mask = jitter_color([1, 0, 0]) if static_color else random_color()
        colors.append(color_mask)
            
        img = np.ones((img_height, img_width, 3)) * color_mask
        colored_mask = np.dstack((img, mask[:, :, 0] * alpha))
        colored_masks.append(colored_mask)

    return np.array(colored_masks), colors


def getNPMasks(masks, shape, alpha=1, static_color=False):
    """
    Generate colored masks and return both the masks and colors.
    
    Args:
        masks: Input masks tensor/array, shape (N, H, W).
        shape: Target shape (H, W).
        alpha: Opacity level.
        static_color: Whether to use static colors.
        
    Returns:
        tuple: (colored_masks, colors) where colors is a list of RGB values.
    """
    if len(masks) == 0:
        return np.zeros((0, shape[0], shape[1], 4)), []
    
    colored_masks = []
    colors = []
    for mask in masks:
        color_mask = jitter_color([1, 0, 0]) if static_color else random_color()
        colors.append(color_mask)

        if mask.ndim == 3:
            mask = mask.squeeze()
        
        img_height, img_width = mask.shape
        img = np.ones((img_height, img_width, 3)) * color_mask
        colored_mask = np.dstack((img, mask * alpha))
        colored_masks.append(colored_mask)

    return np.array(colored_masks), colors


def visualize_coco_anns(coco_api, idx, ax, shape, alpha=1, border_size=5, draw_border=False, static_color=False, border_color="same"):
    if not coco_api: # no instances were detected :(
        return 
    annIds = coco_api.getAnnIds(imgIds=[idx])
    anns = coco_api.loadAnns(annIds)

    masks, colors = getMasks(coco_api, anns, shape, alpha=alpha, static_color=static_color)
    _visualize_masks(ax, masks, draw_border, border_size, colors, border_color=border_color)


def visualize_masks(img, masks, bboxes=None, shape=(512, 512), 
                    alpha=1, border_size=5, border_color='same', 
                    bbox_linewidth=2, draw_border=False, static_color=False, 
                    show_img=False, path=None, figsize=[20, 10], dpi=100):
    """
    Visualizes masks on an image with optional borders and bounding boxes.
    This function resizes the image and masks to the specified shape, applies colors to the masks,
    and optionally draws borders around the masks and bounding boxes with matching colors.

    Args:
        img (torch.Tensor or np.ndarray): Input image tensor or numpy array in shape (H, W) or (H, W, C).
        masks (torch.Tensor): Tensor of masks in shape (N, H, W) or numpy array.
        bboxes (torch.Tensor or np.ndarray, optional): Normalized `(0 to 1)` bounding boxes in cxcywh format, shape (N, 4).
            If provided, these will be drawn on the image.
        shape (tuple): Target shape for resizing the image and masks. Default is (512, 512).
        alpha (float): Opacity level for the masks `(0 to 1)`. Default is 1 (fully opaque).
        border_size (int): Thickness of the border around masks if draw_border is `True`. Default is 5.
        border_color (str): Color of the border. Accepts `['same', 'white']`. Default is `'same'`.
        bbox_linewidth (int): Line width for bounding boxes. Default is 2.
        draw_border (bool): Whether to draw borders around masks. Default is `False`. Default is `False`.
        static_color (bool): If `True`, uses a static color for masks; otherwise, uses random colors. Default is `False`.
        show_img (bool): If `True`, shows the image alongside the masks. Default is `False`.
        path (str, optional): Path to save the visualization. If None, does not save.
        figsize (list): Size of the figure for visualization.
        dpi (int): Dots per inch for the saved figure.
    """
    if isinstance(img, np.ndarray):
        img = torch.tensor(img, dtype=torch.float32)

    img = F.interpolate(img.unsqueeze(0).unsqueeze(0), size=shape, 
                        mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    masks = F.interpolate(masks.unsqueeze(0), size=shape, 
                            mode="bilinear", align_corners=False).squeeze(0)

    if bboxes is not None:
        H, W = img.shape[-2:]
        bboxes = bboxes * torch.tensor([W, H, W, H], dtype=torch.float32)
        bboxes = box_cxcywh_to_xyxy(bboxes)
    
    masks_np, colors = getNPMasks(masks, shape, alpha=alpha, static_color=static_color)

    ncols = 2 if show_img else 1
    fig, ax = plt.subplots(1, ncols, figsize=figsize, dpi=dpi)
    if ncols == 1:
        ax = [ax]
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    ax[0].imshow(img, cmap='gray')
    if show_img:
        ax[1].imshow(img, cmap='gray')
        target_ax = ax[1]
    else:
        target_ax = ax[0]

    if len(masks_np) > 0:
        _visualize_masks(target_ax, masks_np, draw_border, border_size, colors, border_color=border_color)
    _visualize_bboxes(target_ax, bboxes, bbox_linewidth, colors)

    for a in ax:
        a.axis('off')
        a.set_xlim(0, shape[1]-1)
        a.set_ylim(shape[0]-1, 0)

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        
    plt.close(fig)


def save_coco_vis(img, gt_coco, pred_coco, idx, shape, alpha=0.65, 
                  draw_border=True, border_size=5, border_color='same', 
                  static_color=False, show_img=False, path=None):
    """
    Saves a side-by-side visualization of ground truth 
    and predicted COCO annotations for a given image, with an option to show the image separately.

    :param img: Input image tensor or numpy array in shape (H, W) or (H, W, C).
    :param gt_coco: Ground truth COCO annotations.
    :param pred_coco: Predicted COCO annotations.
    :param idx: Index of the image in the COCO dataset.
    :param shape: Target shape for resizing the image.
    :param alpha: Opacity level for the masks (0 to 1). Default is 0.65.
    :param draw_border: Whether to draw borders around masks. Default is True
    :param border_size: Thickness of the border around masks if draw_border is True. Default is 5.
    :param border_color: Color of the border. Accepts ['same', 'white']. Default is 'same'.
    :param static_color: If True, uses a static color for masks; otherwise, uses random colors. Default is False.
    :param path: Path to save the visualization. If None, does not save.
    :param show_img: If True, shows the image alongside the annotations. Default is False
    :return: None
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
        visualize_coco_anns(gt_coco, idx, ax[1], shape=shape, alpha=alpha, 
                            border_size=border_size, draw_border=draw_border, 
                            static_color=static_color, border_color=border_color)

        ax[2].imshow(img, cmap='gray')
        visualize_coco_anns(pred_coco, idx, ax[2], shape=shape, alpha=alpha, 
                            border_size=border_size, draw_border=draw_border, 
                            static_color=static_color, border_color=border_color)

    else:
        fig, ax = plt.subplots(1, 2, figsize=[20, 10])
        fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

        ax[0].imshow(img, cmap='gray')
        visualize_coco_anns(gt_coco, idx, ax[0], shape=shape, alpha=alpha, 
                            border_size=border_size, draw_border=draw_border, 
                            static_color=static_color, border_color=border_color)
    
        ax[1].imshow(img, cmap='gray')
        visualize_coco_anns(pred_coco, idx, ax[1], shape=shape, alpha=alpha, 
                            border_size=border_size, draw_border=draw_border, 
                            static_color=static_color, border_color=border_color)
    
    for a in ax:
        a.axis('off')
        a.set_xlim(0, shape[1])
        a.set_ylim(shape[0]-1, 0)

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, bbox_inches='tight', pad_inches=0)
    
    plt.close(fig)




