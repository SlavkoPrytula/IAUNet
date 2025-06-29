"""
Amodal COCO Metric

This module provides a metric class for amodal instance segmentation evaluation
that extends the standard COCO metric with F1 and AJI scores.
"""

import tempfile
import os.path as osp
from typing import Dict, List, Optional, Sequence, Union
from collections import OrderedDict

from .coco_metric import CocoMetric
from ..api_wrappers.amodal_cocoeval import AmodalCOCOeval
from utils.registry import METRICS


@METRICS.register(name='AmodalCocoMetric')
class AmodalCocoMetric(CocoMetric):
    """
    Amodal COCO evaluation metric.
    
    Extends CocoMetric to include amodal instance segmentation metrics:
    - F1 score
    - AJI (Aggregated Jaccard Index)
    """
    
    default_prefix: Optional[str] = 'amodal_coco'
    
    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'segm',
                 classwise: bool = False,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 sort_categories: bool = False,
                 use_mp_eval: bool = False) -> None:
        """
        Initialize AmodalCocoMetric.
        
        Args:
            include_amodal_metrics: Whether to compute amodal metrics (F1, AJI)
            **kwargs: Arguments passed to parent CocoMetric
        """
        super().__init__(
            ann_file=ann_file,
            metric=metric,
            classwise=classwise,
            proposal_nums=proposal_nums,
            iou_thrs=iou_thrs,
            metric_items=metric_items,
            format_only=format_only,
            outfile_prefix=outfile_prefix,
            file_client_args=file_client_args,
            backend_args=backend_args,
            collect_device=collect_device,
            prefix=prefix,
            sort_categories=sort_categories,
            use_mp_eval=use_mp_eval
        )
        
        self.amodal_coco_eval = None
        
    def compute_metrics(self, results: list) -> Dict[str, float]:
        eval_results = super().compute_metrics(results)
        
        if 'segm' in self.metrics:
            amodal_results = self._compute_amodal_metrics(results)
            eval_results.update(amodal_results)
            
        return eval_results
        
    def _compute_amodal_metrics(self, results: list) -> Dict[str, float]:
        print('Computing amodal metrics (F1, AJI)...')
        
        gts, preds = zip(*results)
        
        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'amodal_results')
        else:
            outfile_prefix = self.outfile_prefix + '_amodal'
            
        try:
            # Setup COCO API if needed
            if self._coco_api is None:
                print('Converting ground truth to coco format for amodal evaluation...')
                coco_json_path = self.gt_to_coco_json(
                    gt_dicts=gts, outfile_prefix=outfile_prefix)
                from ..api_wrappers import COCO
                self._coco_api = COCO(coco_json_path)
                
            # Handle lazy init
            if self.cat_ids is None:
                self.cat_ids = self._coco_api.get_cat_ids(
                    cat_names=self.dataset_meta['classes'])
            if self.img_ids is None:
                self.img_ids = self._coco_api.get_img_ids()
                
            # Convert predictions to coco format
            result_files = self.results2json(preds, outfile_prefix)
            
            amodal_results = OrderedDict()
            
            # Only compute for segmentation
            if 'segm' in result_files:
                try:
                    # Load utility function from coco_metric
                    import json
                    def load(filename):
                        with open(filename, 'r') as file:
                            return json.load(file)
                    
                    predictions = load(result_files['segm'])
                    
                    # Remove bbox from predictions for segmentation evaluation
                    for x in predictions:
                        x.pop('bbox', None)
                        
                    # Create detection results COCO object  
                    coco_dt = self._coco_api.loadRes(predictions)
                    
                    # Create amodal evaluator
                    self.amodal_coco_eval = AmodalCOCOeval(
                        self._coco_api, coco_dt, 'segm')
                        
                    # Set evaluation parameters
                    self.amodal_coco_eval.params.catIds = self.cat_ids
                    self.amodal_coco_eval.params.imgIds = self.img_ids
                    self.amodal_coco_eval.params.maxDets = list(self.proposal_nums)
                    self.amodal_coco_eval.params.iouThrs = self.iou_thrs
                    
                    # Run amodal evaluation
                    self.amodal_coco_eval.evaluate()
                    self.amodal_coco_eval.accumulate()
                    
                    # Get amodal metrics summary
                    amodal_metrics = self.amodal_coco_eval.summarize_amodal()
                    
                    # Add to results with proper prefix
                    for key, value in amodal_metrics.items():
                        amodal_results[f'segm_{key}'] = float(f'{value:.3f}')
                        
                    print(f'Amodal metrics computed:')
                    for key, value in amodal_metrics.items():
                        print(f'  {key}: {value:.3f}')
                        
                except Exception as e:
                    print(f'Error computing amodal metrics: {e}')
                    # Return default values
                    amodal_results.update({
                        'segm_amodal_F1': 0.0,
                        'segm_amodal_precision': 0.0,
                        'segm_amodal_recall': 0.0,
                        'segm_amodal_AJI': 0.0,
                    })
                
                    
        finally:
            if tmp_dir is not None:
                tmp_dir.cleanup()
                
        return amodal_results
