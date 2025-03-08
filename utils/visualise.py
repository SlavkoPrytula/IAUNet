import os
import numpy as np
import torch
from utils.box_ops import box_cxcywh_to_xyxy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches


def visualize(figsize=(30, 30), show_title=True, path='./', cmap='viridis', **images):
    n = len(images)
    plt.figure(figsize=figsize)
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        if show_title:
            plt.title(' '.join(name.split('_')).title().lower())
        plt.imshow(image, cmap=cmap)
    # plt.show()

    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    
    # plt.savefig(path)
    plt.close()


def visualize_grid(figsize=(30, 30), images=None, path='./', rows=1):
    n = len(images) // rows
    plt.figure(figsize=figsize)
    for i in range(len(images)):
        plt.subplot(rows, n, i + 1)
        plt.imshow(images[i, ...])
        plt.xticks([])
        plt.yticks([])
    # plt.show()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()



# def visualize_grid_v2(figsize=(10, 10), masks=None, titles=None, ncols=5, nrows=None, path='./', **params):
#     """
#     Plots a grid of binary masks.

#     Args:
#     - masks (numpy.ndarray): A binary mask array of shape (N, H, W).
#     - ncols (int): The number of columns in the grid.
#     - figsize (tuple): The size of the figure in inches.

#     Returns:
#     - None
#     """
#     N, H, W = masks.shape

#     # Calculate the number of rows in the grid
#     # nrows = (N + ncols - 1) // ncols
#     if not nrows:
#         nrows = ncols
#     # nrows = N // ncols

#     # Create the figure and axes objects
#     fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
#     fig.subplots_adjust(hspace=0.1, wspace=0.1)

#     # Plot each mask in a separate subplot
#     for i, ax in enumerate(axs.flat):
#         if i < N:
#             ax.imshow(masks[i], **params)
#             ax.axis('off')
#             if titles is not None:
#                 ax.set_title(f'{titles[i]}')
#                 # ax.set_title(f'{titles[i]:.4f}')

#     # Remove any unused subplots
#     for i in range(N, nrows*ncols):
#         axs.flat[i].set_visible(False)

#     # Adjust the spacing between subplots
#     fig.tight_layout(pad=0.5)
    
#     plt.savefig(path)
#     plt.close()
    

def visualize_grid_v2(figsize=(10, 10), masks=None, bboxes=None, titles=None, 
                      ncols=5, nrows=None, path='./', **params):
    """
    Plots a grid of binary masks with bounding box annotations.

    Args:
    - masks (numpy.ndarray): A binary mask array of shape (N, H, W).
    - bboxes (numpy.ndarray): A list of bounding boxes for each mask, where each bounding box is (x_min, y_min, x_max, y_max).
    - titles (list): A list of titles for each subplot.
    - ncols (int): The number of columns in the grid.
    - nrows (int): The number of rows in the grid.
    - figsize (tuple): The size of the figure in inches.
    - path (str): The path where the figure is saved.
    """
    N, H, W = masks.shape

    # nrows = (N + ncols - 1) // ncols
    if not nrows:
        nrows = ncols
    # nrows = N // ncols

    # Create the figure and axes objects
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Plot each mask in a separate subplot and add bounding box
    for i, ax in enumerate(axs.flat):
        if i < N:
            ax.imshow(masks[i], **params)
            ax.axis('off')
            if titles is not None:
                ax.set_title(f'{titles[i]}')

            # Draw bounding box
            if bboxes is not None:
                bbox = bboxes[i]
                if isinstance(bbox, np.ndarray):
                    bbox = torch.tensor(bbox)
                bbox = bbox * torch.tensor([W, H, W, H], dtype=torch.float32)
                bbox = box_cxcywh_to_xyxy(bbox)
                x_min, y_min, x_max, y_max = bbox.int().tolist()
                rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     edgecolor='red', facecolor='none', linewidth=2)
                ax.add_patch(rect)
                
    # Remove any unused subplots
    for i in range(N, nrows*ncols):
        axs.flat[i].set_visible(False)

    # Adjust the spacing between subplots
    fig.tight_layout(pad=0.5)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()



def plot_tiled_image(image, fig_size=[10, 10]):
    fig = plt.figure(figsize=fig_size)
    rows = int(np.sqrt(image.shape[0]))
    columns = rows

    for i in range(1, columns*rows+1):
        img = image[i-1, 0, ...]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.axis('off')
    plt.show()
    plt.close()



def plot3d(image, fig_size=[10, 10], path='./', **params):
    # Create 3D coordinates for each pixel
    x = np.arange(0, -image.shape[0], -1)
    y = np.arange(0, image.shape[1], 1)

    xs, ys = np.meshgrid(x, y)
    zs = image
    colors = zs

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(projection='3d')
    plot = ax.scatter(xs, ys, zs, c=colors, **params) #, vmin=zs[zs!=0].min(), vmax=zs[zs!=0].max())
    ax.view_init(elev=45., azim=110)
    ax.set_zlim3d([0.5, 1])

    fig.colorbar(plot, shrink=0.5, aspect=20)

    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()