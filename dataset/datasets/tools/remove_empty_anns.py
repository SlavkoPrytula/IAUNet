import json
import os
from os.path import join, dirname, basename
import pycocotools.mask as mask_util
import numpy as np
from tqdm import tqdm


def remove_empty_annotations(coco_json_path, output_json_path):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    print(len(coco_data['images']))
    # valid_annotations = []
    # skips = 0
    # for ann in coco_data['annotations']:
    #     if ann['segmentation'] and any(seg for seg in ann['segmentation']):
    #         valid_annotations.append(ann)
    #     else:
    #         skips += 1
    raise
    
    valid_image_ids = set(ann['image_id'] for ann in valid_annotations)
    valid_images = [img for img in coco_data['images'] if img['id'] in valid_image_ids]

    coco_data['annotations'] = valid_annotations
    coco_data['images'] = valid_images

    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

    print(f"Filtered COCO JSON saved to {output_json_path}")
    print(f'Skipped {skips} annotations')


if __name__ == "__main__":
    data_root = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/NeurlPS22-CellSeg/coco"
    coco_json_path = join(data_root, "annotations/valid.json")

    output_json_path = join(
        dirname(coco_json_path), 
        basename(coco_json_path).replace('.json', '_filtered.json')
        )

    remove_empty_annotations(coco_json_path, output_json_path)
    print(f"Updated COCO JSON saved to {output_json_path}")
