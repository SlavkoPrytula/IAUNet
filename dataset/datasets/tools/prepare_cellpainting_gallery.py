import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from os.path import join
import json
from datetime import datetime
from skimage import measure
import pycocotools.mask as coco_mask


data_root = '/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/cellpainting-gallery/BR00116991__2020-11-05T19_51_35-Measurement1'
images_dir = os.path.join(data_root, 'Images')
masks_dir_cells = os.path.join(data_root, 'acapella/cells')
masks_dir_nuclei = os.path.join(data_root, 'acapella/nuclei')

df = pd.read_csv(f'{data_root}/2025-01-07_Broad_df_all_w_gt_splits_filtered_pwf1threshold_0.94.csv')



def filter_empty_masks(masks):
    kept_indices = np.any(masks, axis=(0, 1))
    masks = masks[..., kept_indices]
    return masks, kept_indices

def get_masks(mask):
    unique_values = np.unique(mask)
    unique_values = unique_values[unique_values != 0]
    
    instance_masks = []
    for value in unique_values:
        instance_mask = (mask == value).astype(np.uint8)
        instance_masks.append(instance_mask)
    
    if len(instance_masks) > 0:
        instance_masks = np.stack(instance_masks, axis=-1)
    else:
        instance_masks = np.zeros((mask.shape[0], mask.shape[1], 0))

    return instance_masks


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=1):
    polygons = []
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.0)
    
    for contour in contours:
        if contour.shape[0] < 3:  # Filter out too small contours
            raise
        contour = contour - 1  # Adjust contour coordinates to original image size
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance=0)

        if len(contour) < 3:  # Valid polygons with at least 3 points
            continue
            # raise
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        segmentation = [max(0, i) for i in segmentation]  # Ensure no negative values
        polygons.append(segmentation)
    
    return polygons

def create_image_info(image_id, width, height, file_name):
    return {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
        "date_captured": datetime.utcnow().isoformat(' '),
        "license": 1,
        "coco_url": "",
        "flickr_url": ""
    }

def create_annotation_info(annotation_id, image_id, category_id, binary_mask):
    binary_mask_encoded = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    area = coco_mask.area(binary_mask_encoded)
    bbox = coco_mask.toBbox(binary_mask_encoded).tolist()
    segmentation = binary_mask_to_polygon(binary_mask)

    if not segmentation:
        return None
    
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": 0,
        "area": int(area),
        "bbox": bbox,
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    }

def masks2coco(masks, image_id, category_id, ann_id_last, file_name):
    annotations = []
    image_info = create_image_info(image_id, masks.shape[1], masks.shape[0], file_name)
    for i in range(masks.shape[-1]):
        mask = masks[..., i]
        annotation_info = create_annotation_info(ann_id_last, image_id, category_id, mask)
        if annotation_info:
            annotations.append(annotation_info)
            ann_id_last += 1
    return {"images": [image_info], "annotations": annotations}, ann_id_last

def save_image(image, save_path, file_name, extension):
    os.makedirs(save_path, exist_ok=True)
    img_png_path = os.path.join(save_path, os.path.splitext(file_name)[0] + ".png")
    cv2.imwrite(img_png_path, image)
    img_name = os.path.splitext(file_name)[0] + ".png"
    return img_name



def get_image(row_num, col, field_id):
    channels = []
    for channel in [6, 7, 8]:
        image_filename = f"r{row_num:02d}c{col:02d}f{field_id:02d}p01-ch{channel}sk1fk1fl1.tiff"
        image_path = os.path.join(images_dir, image_filename)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        img = cv2.imread(image_path, -1)
        channels.append(img)

    image = np.stack(channels, axis=-1)
    image = (image / np.max(image) * 255).astype(np.uint8)
    # image = cv2.resize(image, (512, 512))
    
    return image


def get_mask(row_num, col, field_id, mask_type):
    mask_filename = f"R{row_num}C{col}_F{field_id}T0P1_Cells_{mask_type}.tiff"
    mask_path = os.path.join(masks_dir_cells if mask_type == "Cell" 
                             else masks_dir_nuclei, mask_filename)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    mask = cv2.imread(mask_path, -1)
    # mask = cv2.resize(mask, (512, 512))
    mask = get_masks(mask)

    return mask



def prepare_coco(df, save_path, category_ids, ann_id_last, split_name):
    """
    Prepares COCO annotations for cell and nuclei categories for a specific split.
    """
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"supercategory": "cell", "id": category_ids["cell"], "name": "cell"},
            {"supercategory": "nucleus", "id": category_ids["nucleus"], "name": "nucleus"},
        ]
    }
    os.makedirs(f"{save_path}/images", exist_ok=True)
    os.makedirs(f"{save_path}/annotations", exist_ok=True)

    skips = 0
    image_id = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        col = int(row['Col'])
        row_num = int(row['Row'])
        field_id = int(row['FieldID'])

        try:
            image = get_image(row_num, col, field_id)
            mask_cell = get_mask(row_num, col, field_id, "Cell")
            mask_nuc = get_mask(row_num, col, field_id, "Nucleus")

            img_file_name = f"R{str(row_num).zfill(2)}C{str(col).zfill(2)}_F{str(field_id).zfill(2)}T0P1.png"
            save_image(image, f"{save_path}/images", img_file_name, extension="png")

            for masks, category_name in zip([mask_cell, mask_nuc], ["cell", "nucleus"]):
                masks, _ = filter_empty_masks(masks)
                if masks.shape[-1] > 0:
                    coco_image, ann_id_last = masks2coco(
                        masks,
                        image_id=image_id,
                        category_id=category_ids[category_name],
                        ann_id_last=ann_id_last,
                        file_name=img_file_name
                    )
                    coco_data["images"].extend(coco_image["images"])
                    coco_data["annotations"].extend(coco_image["annotations"])

            image_id += 1

        except FileNotFoundError as e:
            skips += 1
            print(f"Skipped due to missing file: {e}")

    annotation_file = f"{save_path}/annotations/{split_name}.json"
    with open(annotation_file, "w") as f:
        json.dump(coco_data, f, indent=4)

    print(f"Skipped {skips} images due to missing files.")
    print(f"Saved COCO {split_name} annotations to {annotation_file}.")





if __name__ == "__main__":
    category_ids = {"cell": 1, "nucleus": 2}

    save_path = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/cellpainting-gallery/BR00116991__2020-11-05T19_51_35-Measurement1/coco_v2"

    for split in ["train", "val", "test"]:
    # for split in ["test"]:
    # for split in ["train", "val"]:
        split_df = df[df["setname"] == split]
        
        prepare_coco(
            df=split_df,
            save_path=save_path,
            category_ids=category_ids,
            ann_id_last=1, 
            split_name=split
        )




# import os
# import cv2


# directory = '/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/cellpainting-gallery/BR00116991__2020-11-05T19_51_35-Measurement1/coco/images'


# import os
# import json
# from tqdm import tqdm
# import cv2
# import numpy as np
# from pycocotools.coco import COCO
# from pycocotools import mask as mask_utils


# def validate_coco_json(coco_json_path, images_dir):
#     coco = COCO(coco_json_path)
    
#     print("Validating dataset...")
    
#     errors = []
#     for img_id in tqdm(coco.getImgIds()):
#         img_info = coco.loadImgs(img_id)[0]
#         img_path = os.path.join(images_dir, img_info['file_name'])
        
#         if not os.path.exists(img_path):
#             print(f"Image file not found: {img_path}")
#             raise
        
#         image = cv2.imread(img_path)
#         if image is None:
#             print(f"Failed to load image: {img_path}")
#             raise
        
#         height, width, _ = image.shape
#         if height != img_info['height'] or width != img_info['width']:
#             print(f"Image dimensions mismatch for {img_path}: "
#                           f"expected ({img_info['height']}, {img_info['width']}), "
#                           f"got ({height}, {width})")
#             raise
        
#         ann_ids = coco.getAnnIds(imgIds=img_id)
#         annotations = coco.loadAnns(ann_ids)
        
#         for ann in annotations:
#             if 'segmentation' in ann:
#                 if isinstance(ann['segmentation'], list):  # Polygon mask
#                     rle = mask_utils.frPyObjects(ann['segmentation'], height, width)
#                     binary_mask = mask_utils.decode(rle)
#                 elif isinstance(ann['segmentation'], dict):  # RLE mask
#                     binary_mask = mask_utils.decode(ann['segmentation'])
#                 else:
#                     print(f"Unknown mask format for annotation ID {ann['id']}")
#                     raise
                
#                 if len(binary_mask.shape) != 3:
#                     print(f"Mask shape mismatch for annotation ID {ann['id']}: "
#                                   f"got {binary_mask.shape}")
#                     raise
                
#                 if np.sum(binary_mask) == 0:
#                     print(f"Empty mask for annotation ID {ann['id']}")
#                     raise
    
#     if errors:
#         print("Validation completed with errors:")
#         for err in errors:
#             print(f"- {err}")
#     else:
#         print("All masks and shapes are valid.")

# coco_json_path = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/cellpainting-gallery/BR00116991__2020-11-05T19_51_35-Measurement1/coco_v1/annotations/test.json"
# images_dir = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/cellpainting-gallery/BR00116991__2020-11-05T19_51_35-Measurement1/coco_v1/images"
# validate_coco_json(coco_json_path, images_dir)
