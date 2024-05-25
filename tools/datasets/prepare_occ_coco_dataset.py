import json
import os
from os.path import join
from os import makedirs
from pycocotools.coco import COCO
import numpy as np
from pycocotools import mask as maskUtils


def polygon_to_rle(polygon, height, width):
    """
    Convert polygon annotations to RLE.
    """
    rles = maskUtils.frPyObjects(polygon, height, width)
    rle = maskUtils.merge(rles)
    return rle


# def prepare(ann_file, iou_threshold=0.5):
#     coco = COCO(ann_file)
    
#     dataset = coco.dataset
#     dataset['annotations'] = []

#     dataset['categories'] = [
#         {
#             'id': 1, 
#             'name': 'occluded', 
#             'supercategory': 'none'
#         },
#         # {
#         #     'id': 1, 
#         #     'name': 'not_occluded', 
#         #     'supercategory': 'none'
#         # }
#     ]

#     for img_id in coco.imgs:
#         img_info = coco.loadImgs(img_id)[0]
#         width, height = img_info['width'], img_info['height']
#         ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
#         anns = coco.loadAnns(ann_ids)

#         rles = [polygon_to_rle(ann['segmentation'], height, width) if 'segmentation' in ann else maskUtils.frPyObjects(ann['bbox'], height, width) for ann in anns]

#         for i, ann in enumerate(anns):
#             ious = maskUtils.iou([rles[i]], rles[:i] + rles[i + 1:], [0] * (len(anns) - 1))
#             is_occluded = np.any(ious > iou_threshold)
            
#             ann['category_id'] = 2 if is_occluded else 1
#             dataset['annotations'].append(ann)

#     filepath = os.path.split(ann_file)[0]
#     filename = os.path.splitext(os.path.split(ann_file)[1])[0]
    
#     new_filepath = join(filepath, 'occ')
#     new_filename = filename + '-occ.json'
#     new_ann_file = join(new_filepath, new_filename)

#     makedirs(new_filepath, exist_ok=True)
#     with open(new_ann_file, 'w') as f:
#         json.dump(dataset, f, indent=4)

#     print(f'Modified annotations with new categories saved to: \n- {new_ann_file}')


def prepare(ann_file, iou_threshold=0.5):
    coco = COCO(ann_file)
    
    dataset = coco.dataset
    dataset['annotations'] = []

    dataset['categories'] = [
        {
            'id': 1, 
            'name': 'occluded', 
            'supercategory': 'none'
        }
    ]

    for img_id in coco.imgs:
        img_info = coco.loadImgs(img_id)[0]
        width, height = img_info['width'], img_info['height']
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        rles = [polygon_to_rle(ann['segmentation'], height, width) if 'segmentation' in ann else maskUtils.frPyObjects(ann['bbox'], height, width) for ann in anns]

        for i, ann in enumerate(anns):
            ious = maskUtils.iou([rles[i]], rles[:i] + rles[i + 1:], [0] * (len(anns) - 1))
            is_occluded = np.any(ious > iou_threshold)
            
            if is_occluded:
                ann['category_id'] = 1
                dataset['annotations'].append(ann)

    filepath = os.path.split(ann_file)[0]
    filename = os.path.splitext(os.path.split(ann_file)[1])[0]
    
    new_filepath = join(filepath, 'occ')
    new_filename = filename + '-occ.json'
    new_ann_file = join(new_filepath, new_filename)

    makedirs(new_filepath, exist_ok=True)
    with open(new_ann_file, 'w') as f:
        json.dump(dataset, f, indent=4)

    print(f'Modified annotations with only occluded categories saved to: \n- {new_ann_file}')



# ann_file = '/gpfs/space/home/prytula/data/datasets/synthetic_datasets/rectangle/mixed/rectangles_[valid]_[mixed]_[s=0.1-m=0.6-l=0.1-e=0.2]_[n=100]_[R_min=5_R_max=25]_[overlap=0.0-0.5]_[14.03.24].json'
ann_file = '/gpfs/space/home/prytula/data/datasets/cytoplasm_segmentation/brightfield_v2.0/coco/combined_512x512-upd2/valid.json'
prepare(ann_file, iou_threshold=0.01)
