import sys
sys.path.append('./')

import os.path as osp
import tempfile
import numpy as np
import torch
import json
import pycocotools.mask as mask_util
import random

from unittest import TestCase
from evaluation.mmdet import CocoMetric, AmodalCocoMetric
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pycocotools import mask as mask_util


# --- Dummy config for MMDet evaluator ---
class DummyCfg:
    class model:
        class decoder:
            num_classes = 2
        class evaluator:
            outfile_prefix = None
            metric = 'segm'
            classwise = False
            score_thr = 0.05
            mask_thr = 0.5
            nms_thr = 0.5
    class run:
        save_dir = '.'
        @staticmethod
        def get(x):
            return '.'

# --- Utility functions ---
def generate_random_mask(h, w):
    mask = (np.random.rand(h, w) > 0.5).astype(np.uint8)
    rle = mask_util.encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    return mask, rle

def generate_square_mask(h, w):
    # generate ransom square mask
    mask = np.zeros((h, w), dtype=np.uint8)
    size = random.randint(min(h, w) // 2, min(h, w))
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    mask[y:y+size, x:x+size] = 1
    rle = mask_util.encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    return mask, rle

def generate_full_mask(h, w):
    mask = np.ones((h, w), dtype=np.uint8)
    rle = mask_util.encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    return mask, rle

def generate_empty_mask(h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    rle = mask_util.encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    return mask, rle

# --- Function to create COCO JSON files ---
def make_coco_json(tmp_dir, num_images=10, num_classes=2, h=32, w=32, seed=123, mask_generator=generate_random_mask):
    np.random.seed(seed)
    random.seed(seed)
    images = [{'id': i+1, 'width': w, 'height': h, 'file_name': f'img_{i+1}.jpg'} for i in range(num_images)]
    categories = [{'id': i+1, 'name': f'class_{i+1}'} for i in range(num_classes)]
    # GT annotations
    annotations = []
    gt_masks = {}
    for i in range(num_images):
        mask, rle = mask_generator(h, w)
        gt_masks[i+1] = mask
        cat_id = random.randint(1, num_classes)
        x, y = random.randint(0, w//2), random.randint(0, h//2)
        bw, bh = random.randint(5, w-x), random.randint(5, h-y)
        annotations.append({
            'id': i+1,
            'image_id': i+1,
            'category_id': cat_id,
            'bbox': [x, y, bw, bh],
            'area': float(mask.sum()),
            'iscrowd': 0,
            'segmentation': rle,
        })
    gt_json = {'images': images, 'annotations': annotations, 'categories': categories}
    gt_path = osp.join(tmp_dir, 'gt_rand.json')
    with open(gt_path, 'w') as f:
        json.dump(gt_json, f)
    # Predictions: random class, random bbox, random score, random mask
    preds = []
    pred_masks = {}
    for i in range(num_images):
        mask, rle = mask_generator(h, w)
        pred_masks[i+1] = mask
        cat_id = random.randint(1, num_classes)
        x, y = random.randint(0, w//2), random.randint(0, h//2)
        bw, bh = random.randint(5, w-x), random.randint(5, h-y)
        score = random.uniform(0.5, 1.0)
        preds.append({
            'image_id': i+1,
            'category_id': cat_id,
            'bbox': [x, y, bw, bh],
            'score': score,
            'segmentation': rle,
        })
    pred_path = osp.join(tmp_dir, 'pred_rand.json')
    with open(pred_path, 'w') as f:
        json.dump(preds, f)
    return gt_path, pred_path, images, pred_masks, h, w, preds


# --- Main test class ---
class TestMMDetDataloaderEvaluator(TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        self.tmp_dir.cleanup()

    def _test_segm(self, gt_path, pred_path, images, pred_masks, h, w, preds, metric='segm'):
        # COCOeval reference for segm
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(pred_path)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType=metric)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # CocoMetric
        # coco_metric = CocoMetric(ann_file=gt_path, metric=metric, classwise=True)
        coco_metric = AmodalCocoMetric(
            ann_file=gt_path, 
            metric=metric, 
            classwise=True, 
        )
        categories = coco_metric._coco_api.loadCats(coco_metric._coco_api.getCatIds())
        class_names = [category['name'] for category in categories]
        coco_metric.dataset_meta = dict(classes=class_names)

        for img in images:
            img_id = img['id']
            img_preds = [p for p in preds if p['image_id'] == img_id]

            bboxes = torch.tensor([
                [p['bbox'][0], p['bbox'][1], p['bbox'][0]+p['bbox'][2], p['bbox'][1]+p['bbox'][3]]
                for p in img_preds
            ], dtype=torch.float32)
            scores = torch.tensor([p['score'] for p in img_preds], dtype=torch.float32)
            labels = torch.tensor([p['category_id']-1 for p in img_preds], dtype=torch.int64)
            masks = torch.stack([
                torch.from_numpy(pred_masks[img_id]) for p in img_preds
            ]).to(torch.uint8)

            results = dict()
            results["img_id"] = img_id
            results["ori_shape"] = [h, w]
            results["pred_instances"] = {
                "masks": masks,
                "labels": labels,
                "scores": scores,
                "mask_scores": scores,
                "bboxes": bboxes,
            }

            data_samples = [results]
            coco_metric.process({}, data_samples)

        eval_results = coco_metric.evaluate(len(images))
        print(eval_results)
        raise

        assert abs(coco_eval.stats[0] - eval_results[f'coco/{metric}_mAP']) < 1e-3, \
            f"COCOeval: {coco_eval.stats[0]}, CocoMetric: {eval_results[f'coco/{metric}_mAP']}"
        assert abs(coco_eval.stats[1] - eval_results[f'coco/{metric}_mAP_50']) < 1e-3, \
            f"COCOeval: {coco_eval.stats[1]}, CocoMetric: {eval_results[f'coco/{metric}_mAP_50']}"
        assert abs(coco_eval.stats[2] - eval_results[f'coco/{metric}_mAP_75']) < 1e-3, \
            f"COCOeval: {coco_eval.stats[2]}, CocoMetric: {eval_results[f'coco/{metric}_mAP_75']}"   
            

    def _test_segm_mmdet(self, gt_path, pred_path, images, pred_masks, h, w, preds, metric='segm'):
        # MMDet evaluator
        from evaluation import CocoEvaluator, AmodalCocoEvaluator
        class DummyDataset:
            def __init__(self, ann_file):
                self.ann_file = ann_file
            def __len__(self):
                return len(images)
        cfg = DummyCfg()
        cfg.model.evaluator.metric = metric
        dataset = DummyDataset(gt_path)
        evaluator = AmodalCocoEvaluator(cfg, dataset)
     
        for img in images:
            img_id = img['id']
            img_preds = [p for p in preds if p['image_id'] == img_id]
           
            bboxes = torch.tensor([
                [p['bbox'][0], p['bbox'][1], p['bbox'][0]+p['bbox'][2], p['bbox'][1]+p['bbox'][3]]
                for p in img_preds
            ], dtype=torch.float32)
            scores = torch.tensor([p['score'] for p in img_preds], dtype=torch.float32)
            labels = torch.tensor([p['category_id']-1 for p in img_preds], dtype=torch.int64)
            logits = torch.rand(len(img_preds), evaluator.num_classes + 1)
            masks = torch.stack([
                torch.from_numpy(pred_masks[img_id]) for p in img_preds
            ]).to(torch.uint8)

            preds_dict = {
                'pred_logits': logits.unsqueeze(0),
                'pred_instance_masks': masks.unsqueeze(0),
                'pred_bboxes': bboxes.unsqueeze(0),
                'img_id': [img_id],
                'ori_shape': [(h, w)]
            }
            evaluator.process(preds_dict)
     
        evaluator.evaluate()
        gt_coco = evaluator.gt_coco
        pred_coco = evaluator.pred_coco
        
        coco_eval = COCOeval(gt_coco, pred_coco, iouType=metric)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        assert abs(coco_eval.stats[0] - evaluator.stats[f'{metric}_mAP']) < 1e-3, \
            f"COCOeval: {coco_eval.stats[0]}, CocoMetric: {evaluator.stats[f'{metric}_mAP']}"
        assert abs(coco_eval.stats[1] - evaluator.stats[f'{metric}_mAP_50']) < 1e-3, \
            f"COCOeval: {coco_eval.stats[1]}, CocoMetric: {evaluator.stats[f'{metric}_mAP_50']}"
        assert abs(coco_eval.stats[2] - evaluator.stats[f'{metric}_mAP_75']) < 1e-3, \
            f"COCOeval: {coco_eval.stats[2]}, CocoMetric: {evaluator.stats[f'{metric}_mAP_75']}"   
            
    
    # --- segm tests ---
    def test_random_segm(self):
        gt_path, pred_path, images, pred_masks, h, w, preds = make_coco_json(self.tmp_dir.name, mask_generator=generate_random_mask)
        self._test_segm(gt_path, pred_path, images, pred_masks, h, w, preds, metric='segm')

    def test_square_segm(self):
        gt_path, pred_path, images, pred_masks, h, w, preds = make_coco_json(self.tmp_dir.name, mask_generator=generate_square_mask)
        self._test_segm(gt_path, pred_path, images, pred_masks, h, w, preds, metric='segm')

    def test_full_segm(self):
        gt_path, pred_path, images, pred_masks, h, w, preds = make_coco_json(self.tmp_dir.name, mask_generator=generate_full_mask)
        self._test_segm(gt_path, pred_path, images, pred_masks, h, w, preds, metric='segm')

    def test_empty_segm(self):
        gt_path, pred_path, images, pred_masks, h, w, preds = make_coco_json(self.tmp_dir.name, mask_generator=generate_empty_mask)
        self._test_segm(gt_path, pred_path, images, pred_masks, h, w, preds, metric='segm')

    def test_random_segm_mmdet(self):
        gt_path, pred_path, images, pred_masks, h, w, preds = make_coco_json(self.tmp_dir.name, mask_generator=generate_random_mask)
        self._test_segm_mmdet(gt_path, pred_path, images, pred_masks, h, w, preds, metric='segm')
    
    def test_square_segm_mmdet(self):
        gt_path, pred_path, images, pred_masks, h, w, preds = make_coco_json(self.tmp_dir.name, mask_generator=generate_square_mask)
        self._test_segm_mmdet(gt_path, pred_path, images, pred_masks, h, w, preds, metric='segm')

    def test_full_segm_mmdet(self):
        gt_path, pred_path, images, pred_masks, h, w, preds = make_coco_json(self.tmp_dir.name, mask_generator=generate_full_mask)
        self._test_segm_mmdet(gt_path, pred_path, images, pred_masks, h, w, preds, metric='segm')


    # --- bbox tests ---
    def test_random_bbox(self):
        gt_path, pred_path, images, pred_masks, h, w, preds = make_coco_json(self.tmp_dir.name, mask_generator=generate_random_mask)
        self._test_segm(gt_path, pred_path, images, pred_masks, h, w, preds, metric='bbox')
    
    def test_square_bbox(self):
        gt_path, pred_path, images, pred_masks, h, w, preds = make_coco_json(self.tmp_dir.name, mask_generator=generate_square_mask)
        self._test_segm(gt_path, pred_path, images, pred_masks, h, w, preds, metric='bbox')

    def test_full_bbox(self):
        gt_path, pred_path, images, pred_masks, h, w, preds = make_coco_json(self.tmp_dir.name, mask_generator=generate_full_mask)
        self._test_segm(gt_path, pred_path, images, pred_masks, h, w, preds, metric='bbox')

    def test_empty_bbox(self):
        gt_path, pred_path, images, pred_masks, h, w, preds = make_coco_json(self.tmp_dir.name, mask_generator=generate_empty_mask)
        self._test_segm(gt_path, pred_path, images, pred_masks, h, w, preds, metric='bbox')

    def test_random_bbox_mmdet(self):
        gt_path, pred_path, images, pred_masks, h, w, preds = make_coco_json(self.tmp_dir.name, mask_generator=generate_random_mask)
        self._test_segm_mmdet(gt_path, pred_path, images, pred_masks, h, w, preds, metric='bbox')

    def test_square_bbox_mmdet(self):
        gt_path, pred_path, images, pred_masks, h, w, preds = make_coco_json(self.tmp_dir.name, mask_generator=generate_square_mask)
        self._test_segm_mmdet(gt_path, pred_path, images, pred_masks, h, w, preds, metric='bbox')

    def test_full_bbox_mmdet(self):
        gt_path, pred_path, images, pred_masks, h, w, preds = make_coco_json(self.tmp_dir.name, mask_generator=generate_full_mask)
        self._test_segm_mmdet(gt_path, pred_path, images, pred_masks, h, w, preds, metric='bbox')

    