import json
from pycocotools.coco import COCO
from collections import defaultdict
import numpy as np


def calculate_coco_statistics(json_paths):
    # Initialize cumulative statistics
    cumulative_stats = {
        'num_images': 0,
        'num_annotations': 0,
        'annotations_per_category': defaultdict(int),
        'annotations_per_image': defaultdict(int),
        'area_per_category': defaultdict(list),
        'polygon_points_per_category': defaultdict(list),
    }

    for json_path in json_paths:
        # Load each COCO dataset
        coco = COCO(json_path)
        num_images = len(coco.imgs)
        num_annotations = len(coco.anns)

        # Update cumulative stats
        cumulative_stats['num_images'] += num_images
        cumulative_stats['num_annotations'] += num_annotations

        # Calculate statistics for each dataset
        for ann in coco.anns.values():
            category_id = ann['category_id']
            image_id = ann['image_id']
            area = ann['area']

            # Count annotations per category and image, and area per category
            cumulative_stats['annotations_per_category'][category_id] += 1
            cumulative_stats['annotations_per_image'][image_id] += 1
            cumulative_stats['area_per_category'][category_id].append(area)

            # Calculate number of polygon points per annotation if segmentation is polygon-based
            if isinstance(ann['segmentation'], list):  # Check if segmentation is polygon format
                num_points = sum(len(seg) // 2 for seg in ann['segmentation'])  # Total points in all polygons
                cumulative_stats['polygon_points_per_category'][category_id].append(num_points)

    # Compute derived statistics
    total_images = cumulative_stats['num_images']
    total_annotations = cumulative_stats['num_annotations']
    avg_annotations_per_image = total_annotations / total_images

    category_stats = {}
    for cat_id, count in cumulative_stats['annotations_per_category'].items():
        avg_area = np.mean(cumulative_stats['area_per_category'][cat_id]) if cumulative_stats['area_per_category'][cat_id] else 0
        avg_polygon_points = (
            np.mean(cumulative_stats['polygon_points_per_category'][cat_id]) if cumulative_stats['polygon_points_per_category'][cat_id] else 0
        )
        max_polygon_points = (
            np.max(cumulative_stats['polygon_points_per_category'][cat_id]) if cumulative_stats['polygon_points_per_category'][cat_id] else 0
        )
        
        category_stats[cat_id] = {
            'num_annotations': count,
            'avg_area': avg_area,
            'avg_polygon_points': avg_polygon_points,
            'max_polygon_points': max_polygon_points,
        }

    # Print results
    print(f"Total number of images across all datasets: {total_images}")
    print(f"Total number of annotations across all datasets: {total_annotations}")
    print(f"Average annotations per image: {avg_annotations_per_image:.2f}\n")
    
    print("Annotations per category:")
    for cat_id, cat_stat in category_stats.items():
        category_name = "Unknown"  # Default if category not found
        for json_path in json_paths:
            coco = COCO(json_path)
            if cat_id in coco.cats:
                category_name = coco.cats[cat_id]['name']
                break
        print(f"  - {category_name}:")
        print(f"    - Number of annotations: {cat_stat['num_annotations']}")
        print(f"    - Average area: {cat_stat['avg_area']:.2f}")
        print(f"    - Average polygon points: {cat_stat['avg_polygon_points']:.2f}")
        print(f"    - Maximum polygon points: {cat_stat['max_polygon_points']}\n")
    
    print("Annotations per image (sample):")
    for img_id, count in list(cumulative_stats['annotations_per_image'].items())[:10]:
        print(f"  - Image ID {img_id}: {count} annotations")
    
    return cumulative_stats


json_path = ['/project/project_465001327/datasets/Revvity-25/annotations/train.json', 
             '/project/project_465001327/datasets/Revvity-25/annotations/valid.json']
calculate_coco_statistics(json_path)
