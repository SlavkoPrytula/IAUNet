import json
import os
from os.path import join
import pycocotools.mask as mask_util
import numpy as np
from tqdm import tqdm

def convert_to_rle(segmentation, image_height, image_width):
    """Converts a segmentation to RLE format if it's a polygon."""
    if isinstance(segmentation, list):
        rle = mask_util.frPyObjects(segmentation, image_height, image_width)
        rle = mask_util.merge(rle)
        return rle
    return segmentation

def compute_area_for_annotation(annotation, image_info):
    """Computes the area for a given annotation using the pycocotools."""
    segmentation = annotation['segmentation']
    image_height = image_info['height']
    image_width = image_info['width']
    
    rle = convert_to_rle(segmentation, image_height, image_width)
    
    area = mask_util.area(rle)
    return area

def compute_bbox_from_segmentation(segmentation, image_height, image_width):
    """Computes the bounding box from segmentation in [x_min, y_min, width, height] format."""
    rle = convert_to_rle(segmentation, image_height, image_width)
    mask = mask_util.decode(rle)

    # Find bounding box coordinates
    horizontal_indices = np.where(np.any(mask, axis=0))[0]
    vertical_indices = np.where(np.any(mask, axis=1))[0]

    if horizontal_indices.shape[0] and vertical_indices.shape[0]:
        x_min = int(horizontal_indices[0])
        x_max = int(horizontal_indices[-1])
        y_min = int(vertical_indices[0])
        y_max = int(vertical_indices[-1])
        bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
    else:
        bbox = [0, 0, 0, 0]
    
    return bbox

def update_coco(coco_json_path, output_json_path):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    images_info = {img['id']: img for img in coco_data['images']}
    
    for annotation in tqdm(coco_data['annotations'], desc="Calculating areas and bounding boxes"):
        image_info = images_info[annotation['image_id']]

        area = compute_area_for_annotation(annotation, image_info)
        annotation['area'] = float(area)

        bbox = compute_bbox_from_segmentation(annotation['segmentation'], image_info['height'], image_info['width'])
        annotation['bbox'] = bbox
    
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)


if __name__ == "__main__":
    data_root = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/EVICAN2/coco"
    coco_json_path = join(data_root, "annotations/EVICAN2/processed/instances_eval2019_medium_EVICAN2_cell.json")

    output_json_path = coco_json_path.replace('.json', '.json')

    update_coco(coco_json_path, output_json_path)
    print(f"Updated COCO JSON saved to {output_json_path}")
