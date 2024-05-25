import numpy as np
from .pycococreatortools import create_annotation_info, create_image_info


def get_coco_template():
    # COCO format dictionary
    coco_dict = {
        "info": {
            "year": 2023,
            "version": "1.0",
            "description": "Binary masks converted to COCO format",
            "contributor": "",
            "url": "",
            "date_created": "2023-02-22"
        },
        "licenses": [],
        "categories": [
            {
                "id": 1, 
                "name": "cell",
                "supercategory": ""
            }
            ],
        "images": [],
        "annotations": []
    }

    return coco_dict


def masks2coco(masks: list, scores: list = None):
    # masks shape:  [(N, H, W), ...]
    # scores shape: [(N), ...]

    coco_dict = get_coco_template()

    # Create a new image id
    image_id = 1
    annotation_id = 1

    for i, mask_set in enumerate(masks):
        # mask_set shape: (N, H, W)

        # Create a new image info
        image_info = create_image_info(
            image_id=image_id,
            image_size=mask_set[0].shape,
            file_name=f"image{image_id}.jpg",
            coco_url="",
            date_captured=""
        )

        coco_dict["images"].append(image_info)

        # Loop over every instance mask
        for j, mask in enumerate(mask_set):

            # Create a new annotation info
            annotation_info = create_annotation_info(
                annotation_id=annotation_id,
                image_id=image_id,
                category_info={'is_crowd': 0, 'id': 1},
                binary_mask=mask,
                image_size=mask.shape,
            )

            if annotation_info is None:
                continue

            if scores is not None:
                annotation_info['score'] = scores[i][j]
            else:  # <---
                annotation_info['score'] = 1

            # Append annotation info to COCO dictionary
            coco_dict["annotations"].append(annotation_info)

            # Increment annotation id
            annotation_id += 1

        # Increment image id
        image_id += 1

    return coco_dict
