# import cv2
# import matplotlib.pyplot as plt


# # data_root = '/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/NeurlPS22-CellSeg'
# # img = cv2.imread(f'{data_root}/images/cell_00719.png', -1)
# # mask = cv2.imread(f'{data_root}/labels/cell_00719_label.tiff', -1)
# # print(img.shape, img.min(), img.max(), mask.max())

# # plt.imshow(img)
# # plt.savefig("./test.jpg")







# from os.path import join
# import cv2
# import json
# import numpy as np

# import sys
# sys.path.append(".")

# from utils.coco.coco import COCO 
# from utils.coco.cocoeval import COCOeval 

# from visualizations import save_coco_vis
# from utils.visualise import visualize




#     # NeurlPS22_CellSeg
#     gt_json_path = "gt coco json path"
#     pred_json_path = 'pred coco json path'
#     image_dir = "images path"


#     gt_coco, pred_coco = get_coco(gt_json_path, pred_json_path)


#     image_ids = gt_coco.getImgIds()
    
#     for idx in range(2, 6):
#         img_id = image_ids[idx]
#         img_info = gt_coco.loadImgs(ids=[img_id])[0]
#         img_name = img_info['file_name']
#         base_name = img_name.split(".")[0]
#         img_path = join(image_dir, img_name)

#         img = cv2.imread(img_path, -1)
#         img = img / img.max()

#         if len(img.shape) == 3:
#             img = img[..., 0]

#         H, W = img_info["height"], img_info["width"]
#         save_coco_vis(img, gt_coco, pred_coco, img_id, shape=[H, W], 
#                       path=f"./iclr/{base_name}.jpg")



import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import torch.nn.functional as F
from tqdm import tqdm
import random

import sys
sys.path.append('./')

# from utils.coco.coco import COCO
from pycocotools.coco import COCO
from visualizations.coco_vis import visualize_coco_anns


np.random.seed(3407)
random.seed(3407)


def get_coco(gt_json_path, pred_json_paths):
    """
    Load the ground truth COCO annotations and prediction annotations for each model.
    """
    gt_coco = COCO(gt_json_path)
    pred_cocos = [gt_coco.loadRes(pred_json_path) for pred_json_path in pred_json_paths]
    return gt_coco, pred_cocos


def save_multi_model_vis(imgs, gt_coco, pred_cocos, image_ids, shape, model_names, path=None):
    """
    Save a visualization showing the ground truth and predictions from multiple models for multiple images.
    Each row will correspond to a different image, and each column will correspond to a different model's prediction.
    The last column is always the ground truth (GT).
    """
    num_images = len(image_ids)
    num_models = len(pred_cocos) + 1  # Including GT as the last column

    fig, ax = plt.subplots(num_images, num_models, figsize=[5 * num_models, 5 * num_images])
    fig.subplots_adjust(wspace=0.05, hspace=0.01, left=0, right=1, bottom=0, top=1)  # Small space between columns

    for i, (img, img_id) in tqdm(enumerate(zip(imgs, image_ids)), total=len(image_ids)):
        img = torch.tensor(img, dtype=torch.float32)
        img = F.interpolate(img.unsqueeze(0).unsqueeze(0), size=shape, mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
        img = img.cpu().numpy()

        for j, (pred_coco, model_name) in enumerate(zip(pred_cocos, model_names)):
            ax[i, j].imshow(img, cmap='gray')
            visualize_coco_anns(pred_coco, img_id, ax[i, j], shape=shape, alpha=0.65, draw_border=True, static_color=False)

        ax[i, -1].imshow(img, cmap='gray')
        visualize_coco_anns(gt_coco, img_id, ax[i, -1], shape=shape, alpha=0.65, draw_border=True, static_color=False)

        for a in ax[i, :]:
            a.axis('off')

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=100, bbox_inches='tight', pad_inches=0)

    plt.close(fig)


def visualize_multiple_models(gt_json_path, pred_json_paths, image_dir, num_images=5, save_dir='./output'):
    """
    Visualize predictions from multiple models along with ground truth.
    For each image, display GT and predictions from all models in a row.

    Args:
    - gt_json_path: Path to the ground truth COCO JSON file.
    - pred_json_paths: List of paths to prediction COCO JSON files (for different models).
    - image_dir: Path to the directory containing the images.
    - num_images: Number of images to visualize.
    - save_dir: Directory where the output visualizations will be saved.
    """
    gt_coco, pred_cocos = get_coco(gt_json_path, pred_json_paths)
    image_ids = gt_coco.getImgIds()[:num_images]
    image_ids = [image_ids[0], image_ids[3], image_ids[4]]

    os.makedirs(save_dir, exist_ok=True)

    imgs = []
    for img_id in image_ids:
        img_info = gt_coco.loadImgs(ids=[img_id])[0]
        img_name = img_info['file_name']
        img_path = join(image_dir, img_name)

        img = cv2.imread(img_path, -1)
        img = img / img.max()

        if len(img.shape) == 3:
            img = img[..., 0]

        imgs.append(img)

    H, W = gt_coco.loadImgs(ids=[image_ids[0]])[0]["height"], gt_coco.loadImgs(ids=[image_ids[0]])[0]["width"]

    save_path = f"{save_dir}/multi_image_comparison_livecell_crop_v1.png"
    model_names = [f"Model {i+1}" for i in range(len(pred_json_paths))]

    save_multi_model_vis(imgs, gt_coco, pred_cocos, image_ids, shape=[H, W], model_names=model_names, path=save_path)


# gt_json_path = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/Revvity-25/v2/annotations/valid.json"
# pred_json_paths = [
#     "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/mmdet_results/EXPERIMENTS/EXPERIMENTS_revityy_ext/Revityy/mask-rcnn_r50_fpn_1x_coco/job=52218535/results/coco_test.segm.json",
#     "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/mmdet_results/EXPERIMENTS/EXPERIMENTS_revityy_ext/Revityy/PointRend_r50_caffe_fpn_ms_1x_coco/job=52218539/results/coco_test.segm.json",
#     "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/mmdet_results/EXPERIMENTS/EXPERIMENTS_revityy_ext/Revityy/mask2former_r50_8xb2-lsj-50e_coco/job=52218543/results/coco_valid.segm.json",
#     "runs/benchmarks/[Revvity_25]/[iaunet-r50]/[iadecoder_ml]/[InstanceHead-v2.2.1-dual-update]/[job=52200406]-[2024-09-28 11:07:25]/eval/results/coco.segm.json",
# ]
# image_dir = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/Revvity-25/v2/images"


gt_json_path = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/LiveCell/crop_512x512/coco/annotations/livecell_coco_test.json"
pred_json_paths = [
    "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/mmdet_results/EXPERIMENTS/LiveCell/mask-rcnn_r50_fpn_1x_coco/job=52020558/results/coco_valid.segm.json",
    "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/mmdet_results/EXPERIMENTS/LiveCell/PointRend_r50_caffe_fpn_ms_1x_coco/job=52020557/results/coco_valid.segm.json",
    "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/mmdet_results/EXPERIMENTS/LiveCell/mask2former_r50_8xb2-lsj-50e_coco/job=52020562/results/coco_valid.segm.json",
    "runs/benchmarks/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml]/[InstanceHead-v2.2.1-dual-update]/[job=52108472]-[2024-09-20 10:29:07]/eval/results/coco.segm.json",
]
image_dir = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/LiveCell/crop_512x512/coco/images/livecell_test_images"


visualize_multiple_models(gt_json_path, pred_json_paths, image_dir, 
                          num_images=5, save_dir='./iclr')

