import numpy as np
import datetime
import pycocotools.mask as mask_util

def get_coco_template():
    return {
        "info": {
            "year": "2023",
            "version": "1.0",
            "description": "Exported using ChatGPT's adaptation",
            "contributor": "",
            "url": "",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        },
        "licenses": [],
        "categories": [
            {"id": 1, "name": "object", "supercategory": "object"},
        ],
        "images": [],
        "annotations": []
    }

def create_image_info(image_id, file_name, image_size):
    """Create the image info section for a COCO data point."""
    width, height = image_size
    return {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
        "date_captured": datetime.datetime.utcnow().isoformat(' '),
        "license": 1,
        "coco_url": "",
        "flickr_url": ""
    }

def create_annotation_info(annotation_id, image_id, binary_mask, scores=None, category_id=1):
    """Create the annotation info section for a COCO data point."""
    rle = mask_util.encode(np.asfortranarray(binary_mask))
    rle['counts'] = rle['counts'].decode('utf-8')

    area = np.sum(binary_mask)  # count the number of 1s in the binary mask (true pixels)
    bbox = list(mask_util.toBbox(rle))


    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": rle,
        "area": area,
        "bbox": bbox,
        "iscrowd": 0
    }

    if scores is not None:
        annotation_info['score'] = scores
    else:
        annotation_info['score'] = 1

    return annotation_info

def masks_to_coco_json(masks_list, image_paths, scores_list=None):
    coco_output = get_coco_template()

    annotation_id = 1
    for image_id, (image_path, masks) in enumerate(zip(image_paths, masks_list), start=1):
        image_size = masks.shape[1:] if masks.ndim > 2 else masks.shape  # (H, W) assuming masks could be (N, H, W) or (H, W)
        coco_output["images"].append(create_image_info(image_id, image_path, image_size))

        if masks.ndim > 2:  # Check if masks contain N dimension
            for mask_num, mask in enumerate(masks):
                score = scores_list[image_id-1][mask_num] if scores_list is not None else None
                annotation_info = create_annotation_info(annotation_id, image_id, mask, score)
                coco_output["annotations"].append(annotation_info)
                annotation_id += 1
        else:  # There's only one mask
            score = scores_list[image_id-1] if scores_list is not None else None
            annotation_info = create_annotation_info(annotation_id, image_id, masks, score)
            coco_output["annotations"].append(annotation_info)
            annotation_id += 1

    return coco_output


# Example usage:
# masks = [(np.random.randint(0, 2, (5, 480, 640)) * 255).astype(np.uint8) for _ in range(10)]  # 10 images with 5 masks each
# image_paths = [f"image_{i}.jpg" for i in range(len(masks))]  # Names of the images
# scores_list = [np.random.rand(len(mask_set)) for mask_set in masks]  # Random scores for each mask

# coco_dict = masks_to_coco_json(masks, image_paths, scores_list)
# print(coco_dict)


import sys
sys.path.append("./")

from utils.coco.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.coco.mask2coco import masks2coco

import numpy as np
import matplotlib.pyplot as plt
import time


def flatten_mask(mask, axis=-1):
    f = np.sum(mask, axis)
    f = np.expand_dims(f, axis)
    return f


# create mask 1 of shape (N, H, W)
mask_1 = np.zeros((50, 256, 256))

for i in range(50):
    x = np.random.randint(0, 256)
    y = np.random.randint(0, 256)
    w = np.random.randint(1, 40)
    h = np.random.randint(1, 40)
    
    mask_1[i, y:y+h, x:x+w] = 1


mask_2 = mask_1.copy()


# prepare masks
gt_masks = [mask_1.astype(np.uint8)]
pred_masks = [mask_2.astype(np.uint8)]

start_time = time.time()

# masks -> coco
image_paths = [f"image_{i}.jpg" for i in range(len(gt_masks))]
gt_coco = masks_to_coco_json(gt_masks, image_paths)
pred_coco = masks_to_coco_json(pred_masks, image_paths)

end_time = time.time()

# load coco dict
gt_coco = COCO(gt_coco)
pred_coco = COCO(pred_coco)
coco_eval = COCOeval(gt_coco, pred_coco, iouType='segm')
coco_eval.params.maxDets = [100, 100, 100]

# Run the evaluation
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()