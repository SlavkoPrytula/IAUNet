import sys
from os.path import join
from os import listdir
from pathlib import Path

sys.path.append("./")

from tools.coco_metric.json_coco_metric import json_coco_evaluation


mmdet_path = "/gpfs/space/home/prytula/scripts/experimental_segmentation/mmdetection/mmdetection"
experiment_path = "work_dirs/brightfield_coco_v2/mask-rcnn_r50_fpn_1x_coco/job=50320803"
experiment_path = join(mmdet_path, experiment_path)

gt_json_path = '/gpfs/space/home/prytula/data/datasets/cytoplasm_segmentation/brightfield_v2.0/coco/train-v1_valid-v2-upd2/valid.json'


evaluation_results = []
for run_dir in sorted(Path(experiment_path).glob('run=*')):
    print(f"Runnign evaluation on:\n- {run_dir}")

    pred_json_path = join(run_dir, 'results', 'coco_eval.segm.json')

    gt_coco, pred_coco, results = json_coco_evaluation(gt_json_path, pred_json_path, return_stats=True)
    print(results)
    evaluation_results.append(results)


sum_metrics = {key: 0 for key in evaluation_results[0].keys()}
for result in evaluation_results:
    for key, value in result.items():
        sum_metrics[key] += value

average_metrics = {key: value / len(evaluation_results) for key, value in sum_metrics.items()}

for key, value in average_metrics.items():
    print(f"{key}: {value:.3f}")
