import json
import os
from os.path import join, dirname, basename
import pycocotools.mask as mask_util
import numpy as np
from tqdm import tqdm


def remove_label_annotations(coco_json_path, output_json_path, label_ids):
    with open(coco_json_path, 'r') as file:
        coco_data = json.load(file)
    
    _annotations = [ann for ann in tqdm(coco_data['annotations']) 
                    if ann['category_id'] not in label_ids]
    coco_data['annotations'] = _annotations

    _categories = [cat for cat in coco_data['categories'] 
                           if cat['id'] not in label_ids]
    coco_data['categories'] = _categories
    
    with open(output_json_path, 'w') as file:
        json.dump(coco_data, file, indent=4)

    print(f"Annotations with category IDs {label_ids} removed and saved to {output_json_path}")



if __name__ == "__main__":
    data_root = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/cellpainting-gallery/BR00116991__2020-11-05T19_51_35-Measurement1/coco_v1"
    coco_json_path = join(data_root, "annotations/val.json")

    output_json_path = join(
        dirname(coco_json_path), 
        basename(coco_json_path).replace('val.json', 'val_cell.json')
        )

    remove_label_annotations(coco_json_path, output_json_path, label_ids=[2])
    print(f"Updated COCO JSON saved to {output_json_path}")
