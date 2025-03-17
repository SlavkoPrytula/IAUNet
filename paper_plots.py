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
    # image_ids = image_ids[5:]

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




gt_json_path = "/project/project_465001327/datasets/Revvity-25/annotations/valid.json"
pred_json_paths = {
    "IAUNet": "runs/benchmarks_v2/[Revvity_25]/[iaunet-r50]/[iadecoder_ml_fpn]/[experimental]/[deep_supervision]/[job=9865327]-[2025-03-10 20:19:46]/eval/results/coco.segm.json",
    "IAUNet": "runs/benchmarks_v2/[Revvity_25]/[iaunet-r50]/[iadecoder_ml_fpn]/[experimental]/[deep_supervision]/[job=9865327]-[2025-03-10 20:19:46]/eval/results/coco.segm.json",
}
image_dir = "/project/project_465001327/datasets/Revvity-25/images"

visualize_multiple_models(gt_json_path, pred_json_paths, image_dir, 
                          num_images=2, dpi=100,
                          save_dir='./cvpr', save_name='multi_image_comparison_revvity_25')
