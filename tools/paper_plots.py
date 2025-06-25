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

from utils.coco.boundary_iou.coco_instance_api.coco import COCO
from utils.coco.boundary_iou.coco_instance_api.cocoeval import COCOeval
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


def compute_metrics(gt_coco, pred_coco, image_id):
    """Compute COCO metrics for a single image."""

    # temp gt coco.
    gt_anns = gt_coco.imgToAnns[image_id]
    gt_json = {
        "images": [gt_coco.loadImgs(image_id)[0]], 
        "annotations": gt_anns, 
        "categories": gt_coco.dataset["categories"]
        }
    temp_gt_coco = COCO()
    temp_gt_coco.dataset = gt_json
    temp_gt_coco.createIndex()

    # temp pred coco.
    pred_anns = pred_coco.imgToAnns[image_id]
    # shift category ids.
    for i in range(len(pred_anns)):
        pred_anns[i]['category_id'] += gt_coco.dataset["categories"][0]['id'] - pred_anns[i]['category_id']
    pred_json = {
        "images": [gt_coco.loadImgs(image_id)[0]], 
        "annotations": pred_anns, 
        "categories": gt_coco.dataset["categories"]
        }
    temp_pred_coco = COCO()
    temp_pred_coco.dataset = pred_json
    temp_pred_coco.createIndex()

    # eval.
    coco_eval = COCOeval(temp_gt_coco, temp_pred_coco, iouType="segm")
    # coco_eval = COCOeval(temp_gt_coco, temp_pred_coco, iouType="boundary")
    coco_eval.params.imgIds = [image_id]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = {
        "mAP@0.5:0.95": coco_eval.stats[0],
        "mAP@0.5": coco_eval.stats[1],
        "mAP@0.75": coco_eval.stats[2],
        "mAP(s)@0.5": coco_eval.stats[3],
        "mAP(m)@0.5": coco_eval.stats[4],
        "mAP(l)@0.5": coco_eval.stats[5],
    }
    return stats



def save_multi_model_vis(imgs, gt_coco, pred_cocos, image_ids, shape, model_names, dpi=100, path=None):
    """
    Save a visualization showing the ground truth and predictions from multiple models for multiple images.
    Each row will correspond to a different image, and each column will correspond to a different model's prediction.
    The last column is always the ground truth (GT).
    """
    num_images = len(image_ids)
    num_models = len(pred_cocos) + 1  # Including GT as the last column

    fig, ax = plt.subplots(num_images, num_models, figsize=[5 * num_models, 5 * num_images])
    fig.subplots_adjust(wspace=0.05, hspace=0.01, left=0, right=1, bottom=0, top=1)  # Small space between columns

    # flatten ax if only one row or one column.
    if num_images == 1:
        ax = np.atleast_2d(ax).reshape(1, -1)
    elif num_models == 1:
        ax = np.atleast_2d(ax).reshape(-1, 1)

    # Add titles for each column
    # model_names = ["MaskRCNN-R50", "PointRend-R50", "Mask2Former-R50", "IAUNet-R50 (ours)"]
    # model_names = ["MaskDINO", "IAUNet-R50 (ours)"]
    for j, model_name in enumerate(model_names + ["GT"]):
        ax[0, j].set_title(model_name, fontsize=28, pad=20)

    for i, (img, img_id) in tqdm(enumerate(zip(imgs, image_ids)), total=len(image_ids)):
        img = torch.tensor(img, dtype=torch.float32)
        img = F.interpolate(img.unsqueeze(0).unsqueeze(0), size=shape, 
                            mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
        img = img.cpu().numpy()

        # visualize predictions.
        for j, (pred_coco, model_name) in enumerate(zip(pred_cocos, model_names)):
            ax[i, j].imshow(img, cmap='gray')
            visualize_coco_anns(pred_coco, img_id, ax[i, j], shape=shape, border_size=5,
                                alpha=0.65, draw_border=True, static_color=False)
        
            # >>>> compute metrics per image <<<<
            metrics = compute_metrics(gt_coco, pred_coco, img_id)
            ax[i, j].text(0.05, 0.95, rf"$\text{{mAP}} = {metrics['mAP@0.5:0.95']*100:.1f}$",
                          color='white', ha='left', va='top', transform=ax[i, j].transAxes,
                          fontsize=28, weight='bold',
                          bbox=dict(
                                facecolor='#2b2b2b', alpha=0.8,
                                edgecolor='white', linewidth=2,
                                boxstyle='round,pad=0.25'
                            ))

        # visualize gt.
        ax[i, -1].imshow(img, cmap='gray')
        visualize_coco_anns(gt_coco, img_id, ax[i, -1], shape=shape, border_size=5,
                            alpha=0.65, draw_border=True, static_color=False)

        for a in ax[i, :]:
            a.axis('off')

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)

    plt.close(fig)



def visualize_multiple_models(gt_json_path, pred_json_paths, image_dir, num_images=5, dpi=100,
                              save_dir='./output', save_name='multi_image_comparison.png'):
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
    # get gt coco and a list of pred cocos.
    _pred_json_paths = list(pred_json_paths.values())
    gt_coco, pred_cocos = get_coco(gt_json_path, _pred_json_paths)
    image_ids = gt_coco.getImgIds()[:num_images]
    # image_ids = [image_ids[0], image_ids[3], image_ids[4]]
    # image_ids = [image_ids[5], image_ids[6], image_ids[7]]
    # image_ids = image_ids[:5]

    # Revvity-25
    image_ids = [image_ids[3], image_ids[4], image_ids[5]]
    
    # LiveCell
    # image_ids = [image_ids[0], image_ids[3], image_ids[11]]
    
    # ISBI2014
    # image_ids = [image_ids[12], image_ids[18], image_ids[11]]

    # metrics_per_image = compute_metrics(gt_coco, pred_cocos[0], image_ids[0])
    # print(metrics_per_image)
    # raise

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

    save_path = f"{save_dir}/{save_name}.png"
    model_names = list(pred_json_paths.keys())

    save_multi_model_vis(imgs, gt_coco, pred_cocos, image_ids, shape=[H, W], dpi=dpi, 
                         model_names=model_names, path=save_path)



# LiveCell
# gt_json_path = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/LiveCell/crop_512x512/coco/annotations/livecell_coco_test.json"
# pred_json_paths = {
#     "MaskRCNN-R50": "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/mmdet_results/EXPERIMENTS/LiveCell/mask-rcnn_r50_fpn_1x_coco/job=52020558/results/coco_valid.segm.json",
#     "PointRend-R50": "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/mmdet_results/EXPERIMENTS/LiveCell/PointRend_r50_caffe_fpn_ms_1x_coco/job=52020557/results/coco_valid.segm.json",
#     "Mask2Former-R50": "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/mmdet_results/EXPERIMENTS/LiveCell/mask2former_r50_8xb2-lsj-50e_coco/job=52020562/results/coco_valid.segm.json",
#     "MaskDINO-R50": "/gpfs/helios/home/prytula/scripts/experimental_segmentation/MaskDINO/runs/livecell_crop/maskdino_R50/[job=]-[2025-03-06_14-32-36]/inference/coco_instances_results.json",
#     "IAUNet-R50 (ours)": "temp/runs/benchmarks_v2/[LiveCellCrop]/[iaunet-r50]/coco.segm.json",
# }
# image_dir = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/LiveCell/crop_512x512/coco/images/livecell_test_images"


# Revvity-25
gt_json_path = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/Revvity-25/v2/annotations/valid.json"
pred_json_paths = {
    "MaskRCNN-R50": "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/mmdet_results/EXPERIMENTS/EXPERIMENTS_revityy_ext/Revityy/mask-rcnn_r50_fpn_1x_coco/job=52218535/results/coco_test.segm.json",
    "PointRend-R50": "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/mmdet_results/EXPERIMENTS/EXPERIMENTS_revityy_ext/Revityy/PointRend_r50_caffe_fpn_ms_1x_coco/job=52218539/results/coco_test.segm.json",
    "Mask2Former-R50": "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/mmdet_results/EXPERIMENTS/EXPERIMENTS_revityy_ext/Revityy/mask2former_r50_8xb2-lsj-50e_coco/job=52218543/results/coco_valid.segm.json",
    "MaskDINO-R50": "/gpfs/helios/home/prytula/scripts/experimental_segmentation/MaskDINO/runs/revvity_25/maskdino_R50/[job=]-[2025-03-09_15-18-07]/inference/coco_instances_results.json",
    "IAUNet-R50 (ours)": "temp/runs/benchmarks_v2/[Revvity_25]/[iaunet-r50]/coco.segm.json",
}
image_dir = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/Revvity-25/v2/images"


# ISBI2014
# gt_json_path = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/ISBI2014/coco/annotations/isbi_test.json"
# pred_json_paths = {
#     "MaskRCNN-R50": "/gpfs/space/home/prytula/scripts/experimental_segmentation/mmdetection/mmdetection/work_dirs/experiments/original_configs/ISBI2014/mask-rcnn_r50_fpn_1x_coco/[job=54056331]/results/coco_valid.segm.json",
#     "PointRend-R50": "/gpfs/space/home/prytula/scripts/experimental_segmentation/mmdetection/mmdetection/work_dirs/experiments/original_configs/ISBI2014/point_rend_r50_fpn_1x_coco/[job=54055617]/results/coco_valid.segm.json",
#     "Mask2Former-R50": "/gpfs/space/home/prytula/scripts/experimental_segmentation/mmdetection/mmdetection/work_dirs/experiments/original_configs/ISBI2014/mask2former_r50_8xb2-lsj-50e_coco/[job=54056352]/results/coco_valid.segm.json",
#     "MaskDINO-R50": "/gpfs/helios/home/prytula/scripts/experimental_segmentation/MaskDINO/runs/isbi2014/maskdino_R50/[job=]-[2025-03-06_12-42-19]/inference/coco_instances_results.json",
#     "IAUNet-R50 (ours)": "temp/runs/benchmarks_v2/[ISBI2014]/[iaunet-r50]/coco.segm.json",
# }
# image_dir = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/ISBI2014/coco/isbi_test"


# ============================
# specialized models - LiveCell
# gt_json_path = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/LiveCell/crop_512x512/coco/annotations/livecell_coco_test.json"
# pred_json_paths = {
#     "CellPose": "/gpfs/space/home/prytula/scripts/experimental_segmentation/cellpose/runs/LiveCellCrop/[job=52054076]/results/coco_eval.segm.json",
#     "CellPose + SM": "/gpfs/space/home/prytula/scripts/experimental_segmentation/cellpose/runs/LiveCellCrop/[job=52070100]/results/coco_eval.segm.json",
#     "IAUNet-R50 (ours)": "temp/runs/benchmarks_v2/[LiveCellCrop]/[iaunet-r50]/coco.segm.json",
# }
# image_dir = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/LiveCell/crop_512x512/coco/images/livecell_test_images"


# specialized models - ISBI2014
# gt_json_path = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/ISBI2014/coco/annotations/isbi_test.json"
# pred_json_paths = {
#     "CellPose": "/gpfs/space/home/prytula/scripts/experimental_segmentation/cellpose/runs/ISBI2014_Cell/[job=54062579]/results/coco_eval_combined.segm.json",
#     "CellPose + SM": "/gpfs/space/home/prytula/scripts/experimental_segmentation/cellpose/runs/ISBI2014_Cell/[job=54062579]/results/coco_eval_sm_combined.segm.json",
#     "IAUNet-R50 (ours)": "temp/runs/benchmarks_v2/[ISBI2014]/[iaunet-r50]/coco.segm.json",
# }
# image_dir = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/ISBI2014/coco/isbi_test"


visualize_multiple_models(gt_json_path, pred_json_paths, image_dir, 
                          num_images=20, dpi=100,
                          save_dir='./cvpr', 
                        #   save_name='multi_model_comparison_isbi2014_v1_specialized_models',
                        #   save_name='multi_model_comparison_livecell_crop_v2'
                          save_name='multi_model_comparison_revvity_25_v3'
                          )

