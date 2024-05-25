from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from os.path import join
import cv2

import sys
sys.path.append(".")

from visualizations import save_coco_vis


def load_coco_json(json_file_path):
    coco = COCO(json_file_path)
    return coco

def json_coco_evaluation(gt_json_path, pred_json_path, return_stats=False):
    coco_gt = load_coco_json(gt_json_path)
    coco_pred = coco_gt.loadRes(pred_json_path)
    
    coco_eval = COCOeval(coco_gt, coco_pred, 'segm')
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
    return coco_gt, coco_pred


if __name__ == "__main__":
    # gt_json_path = '/gpfs/space/home/prytula/data/datasets/cytoplasm_segmentation/brightfield_v2.0/coco/combined_512x512-upd2/valid.json'
    # pred_json_path = '/gpfs/space/home/prytula/scripts/experimental_segmentation/mmdetection/mmdetection/work_dirs/coco_detection/test.segm.json'

    # gt_json_path = '/gpfs/space/home/prytula/data/datasets/cytoplasm_segmentation/brightfield_v2.0/coco/combined_512x512-upd2/valid.json'
    # pred_json_path = "runs/evals/[iaunet]/[brightfield_coco_v2.0]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=50319950]-[2024-02-10 17:51:41]/base/results/coco.segm.json"

    # gt_json_path = '/gpfs/space/home/prytula/data/datasets/cytoplasm_segmentation/brightfield_v2.0/coco/train-v1_valid-v2-upd2/valid.json'
    # pred_json_path = "/gpfs/space/home/prytula/scripts/experimental_segmentation/mmdetection/mmdetection/work_dirs/brightfield_coco_v2/mask-rcnn_r50_fpn_1x_coco/job=50320803/run=1/results/coco_eval.segm.json"
    # pred_json_path = "runs/[iaunet]/[brightfield_coco_v2.0]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=50320116]-[2024-02-11 01:15:49]/results/coco.segm.json"

    # gt_json_path = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/LiveCell/crop_512x512/coco/annotations/livecell_coco_test.json"
    # pred_json_path = "/gpfs/space/home/prytula/scripts/experimental_segmentation/yolo/ultralytics/runs/LiveCell/yolo/yolov8x-seg\/results/coco.segm.json"
    
    gt_json_path = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/synthetic_datasets/worms/mixed/coco/worms_[valid]_[max_s=3]_[min_l=0.01_max_l=0.5]_[min_t=30_max_t=30]_[n=1000]_[R_min=1_R_max=25]_[25.04.24].json"
    pred_json_path = "/gpfs/space/home/prytula/scripts/experimental_segmentation/mmdetection/mmdetection/work_dirs_benchmarks/worms/mask-rcnn_r50_fpn_1x_coco/job=51069259/run=1/results/coco_eval.segm.json"
    # pred_json_path = "runs/[resnet_iaunet_multitask]/[worms]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[base]/[job=51037677]-[2024-04-25 16:56:09]/results/coco.segm.json"


    gt_coco, pred_coco = json_coco_evaluation(gt_json_path, pred_json_path)


    # plot results
    # image_dir = "/gpfs/space/home/prytula/data/datasets/cytoplasm_segmentation/brightfield_v2.0/coco/combined_512x512-upd2/images/valid"
    # image_dir = "/gpfs/space/home/prytula/data/datasets/cytoplasm_segmentation/brightfield_v2.0/coco/train-v1_valid-v2-upd2/images/valid"
    # image_dir = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/LiveCell/crop_512x512/coco/images/livecell_test_images"
    image_dir = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/synthetic_datasets/worms/mixed/coco/images/worms_[valid]_[max_s=3]_[min_l=0.01_max_l=0.5]_[min_t=30_max_t=30]_[n=1000]_[R_min=1_R_max=25]_[25.04.24]"

    # idx = 101438
    
    for idx in range(1, 6):
        img_info = gt_coco.loadImgs(ids=[idx])[0]
        img_name = img_info['file_name']
        base_name = img_name.split(".")[0]
        img_path = join(image_dir, img_name)

        img = cv2.imread(img_path, -1)#[..., 0]
        img = img / img.max()

        H, W = img_info["height"], img_info["width"]
        save_coco_vis(img, gt_coco, pred_coco, idx, shape=[H, W], path=f"./tools/coco_metric/results/{base_name}_iaunet.jpg")