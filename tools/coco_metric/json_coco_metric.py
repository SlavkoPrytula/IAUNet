from os.path import join
import cv2
import json
import numpy as np

import sys
sys.path.append(".")

# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval

# from utils.coco.coco import COCO 
from utils.coco.cocoeval import COCOeval 

from utils.coco.api_wrappers import COCO 
from utils.coco.api_wrappers import COCOeval 

from visualizations import save_coco_vis
from visualizations.visualise import visualize


def load_coco_json(json_file_path):
    coco = COCO(annotation_file=json_file_path)
    return coco

def extract_image_ids(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    image_ids = {item['image_id'] for item in data}
    return image_ids

def compare_image_ids(gt_json_path, pred_json_path):
    coco_gt = load_coco_json(gt_json_path)
    gt_image_ids = set(coco_gt.getImgIds())
    
    pred_image_ids = extract_image_ids(pred_json_path)

    num_gt_images = len(gt_image_ids)
    num_pred_images = len(pred_image_ids)

    print(f"Number of unique images in GT: {num_gt_images}")
    print(f"Number of unique images in Predictions: {num_pred_images}")

    if gt_image_ids == pred_image_ids:
        print("Image IDs match between GT and Predictions.")
    else:
        print("Image IDs do NOT match between GT and Predictions.")
        missing_in_gt = pred_image_ids - gt_image_ids
        missing_in_pred = gt_image_ids - pred_image_ids
        
        if missing_in_gt:
            print(f"Image IDs in Predictions but not in GT: {missing_in_gt}")
        if missing_in_pred:
            print(f"Image IDs in GT but not in Predictions: {missing_in_pred}")

def load(filename):
    print(f"loading from: {filename}")
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def json_coco_evaluation(gt_json_path, pred_json_path, return_stats=False):
    compare_image_ids(gt_json_path, pred_json_path)
    coco_gt = load_coco_json(gt_json_path)

    predictions = load(pred_json_path)

    for pred in predictions:
        # pred['category_id'] += 1 # fix
        if 'bbox' in pred:
            del pred['bbox']

    coco_pred = coco_gt.loadRes(predictions)
    
    coco_eval = COCOeval(coco_gt, coco_pred, 'segm')
    # coco_eval = COCOeval(coco_gt, coco_pred, 'boundary')
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    if return_stats:
        key_mapping = [
            "mAP@0.5:0.95",
            "mAP@0.5",
            "mAP@0.75",
            "mAP(s)@0.5",
            "mAP(m)@0.5",
            "mAP(l)@0.5",
        ]
        stats = coco_eval.stats
        stats = dict(zip(key_mapping, stats))

        return coco_gt, coco_pred, stats
    return coco_gt, coco_pred, None


if __name__ == "__main__":
    # gt_json_path = '/gpfs/space/home/prytula/data/datasets/cytoplasm_segmentation/brightfield_v2.0/coco/combined_512x512-upd2/valid.json'
    # pred_json_path = '/gpfs/space/home/prytula/scripts/experimental_segmentation/mmdetection/mmdetection/work_dirs/coco_detection/test.segm.json'

    # gt_json_path = '/gpfs/space/home/prytula/data/datasets/cytoplasm_segmentation/brightfield_v2.0/coco/combined_512x512-upd2/valid.json'
    # pred_json_path = "runs/evals/[iaunet]/[brightfield_coco_v2.0]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=50319950]-[2024-02-10 17:51:41]/base/results/coco.segm.json"

    # gt_json_path = '/gpfs/space/home/prytula/data/datasets/cytoplasm_segmentation/brightfield_v2.0/coco/train-v1_valid-v2-upd2/valid.json'
    # pred_json_path = "/gpfs/space/home/prytula/scripts/experimental_segmentation/mmdetection/mmdetection/work_dirs/brightfield_coco_v2/mask-rcnn_r50_fpn_1x_coco/job=50320803/run=1/results/coco_eval.segm.json"
    # pred_json_path = "runs/[iaunet]/[brightfield_coco_v2.0]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=50320116]-[2024-02-11 01:15:49]/results/coco.segm.json"

    # gt_json_path = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/LiveCell/crop_512x512/coco/annotations/livecell_coco_test.json"
    # pred_json_path = "runs/[resnet_iaunet_multitask_ml]/[truncated_decoder-iadecoder_ml]/[ResNet]/[LiveCellCrop]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[InstanceHead-v2.2.1-dual-update]/[job=51978235]-[2024-08-31 21:43:39]/eval/results/coco.segm.json"
    
    # gt_json_path = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/EVICAN2/coco/annotations/EVICAN2/processed/instances_eval2019_easy_EVICAN2_cell.json"
    # pred_json_path = "/gpfs/space/home/prytula/scripts/experimental_segmentation/yolo/ultralytics/runs/EVICAN2_Easy/yolo/yolov8m-seg/run2/results/coco_eval.segm.json"
    # image_dir = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/EVICAN2/coco/images/EVICAN_eval2019"

    
    # gt_json_path = "/project/project_465001327/datasets/LiveCell/crop_512x512/coco/annotations/livecell_coco_test.json"
    # pred_json_path = '/gpfs/helios/home/prytula/scripts/experimental_segmentation/MaskDINO/runs/livecell_crop/maskdino_R50/[job=]-[2025-02-17_15-57-40]/inference/coco_instances_results.json'
    # image_dir = "/project/project_465001327/datasets/LiveCell/crop_512x512/coco/images/livecell_test_images"
    
    # Revvity-25
    gt_json_path = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/Revvity-25/v2/annotations/valid.json"
    pred_json_path = 'runs/benchmarks_v2/[Revvity_25]/[iaunet-r50]/[iadecoder_ml_fpn]/[experimental]/[deep_supervision]/[job=58405935]-[2025-07-04 20:21:57]/results/coco.segm.json'
    image_dir = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/Revvity-25/v2/images"

    # ISBI2014
    # gt_json_path = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/ISBI2014/coco/annotations/isbi_test.json"
    # pred_json_path = '/gpfs/helios/home/prytula/scripts/experimental_segmentation/Cell-DETR/runs/[job=53430118]-[2025-02-02 23:37:31]/results/coco_eval.segm.json'
    # image_dir = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/ISBI2014/coco/isbi_test"

    gt_coco, pred_coco, _ = json_coco_evaluation(gt_json_path, pred_json_path)


    # # idx = 101438
    image_ids = gt_coco.getImgIds()
    
    for idx in range(2, 8):
        img_id = image_ids[idx]
        img_info = gt_coco.loadImgs(ids=[img_id])[0]
        img_name = img_info['file_name']
        base_name = img_name.split(".")[0]
        img_path = join(image_dir, img_name)

        img = cv2.imread(img_path, -1)
        img = img / img.max()

        if len(img.shape) == 3:
            img = img[..., 0]

        H, W = img_info["height"], img_info["width"]
        save_coco_vis(img, gt_coco, pred_coco, img_id, shape=[H, W],
                      alpha=0.65, draw_border=True, border_size=5, border_color='white',
                      static_color=False, show_img=False, 
                      path=f"./tools/coco_metric/results/{base_name}.jpg")
                      