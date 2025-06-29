"""
Amodal COCO Evaluator

This evaluator extends the standard CocoEvaluator to support amodal instance 
segmentation evaluation with F1 and AJI metrics.
"""

from os.path import join
from configs import cfg

from .coco_evaluator import CocoEvaluator
from utils.registry import EVALUATORS
from evaluation.mmdet.metrics import AmodalCocoMetric


@EVALUATORS.register(name="AmodalCocoEvaluator")
class AmodalCocoEvaluator(CocoEvaluator):
    """
    Amodal COCO Evaluator for instance segmentation.
    
    Extends CocoEvaluator to include amodal metrics:
    - F1 score
    - AJI (Aggregated Jaccard Index)  
    """

    key_mapping = {
        # amodal metrics
        'amodal_coco/segm_amodal_F1': 'amodal_F1',
        'amodal_coco/segm_amodal_precision': 'amodal_precision',
        'amodal_coco/segm_amodal_recall': 'amodal_recall',
        'amodal_coco/segm_amodal_AJI': 'amodal_AJI',
    }
    
    def __init__(self, cfg: cfg, dataset=None, **kwargs):
        super(CocoEvaluator, self).__init__(cfg, **kwargs)
        
        self.dataset = dataset
        outfile_prefix = cfg.model.evaluator.outfile_prefix
        self.num_classes = cfg.model.decoder.num_classes
        self.metric_type = cfg.model.evaluator.metric
        
        print(f"Doing amodal evaluation on dataset.ann_file: {dataset.ann_file}")
        
        self.metric = AmodalCocoMetric(
            ann_file=dataset.ann_file,
            metric=cfg.model.evaluator.metric,
            classwise=cfg.model.evaluator.classwise,
            outfile_prefix=join(cfg.run.save_dir, outfile_prefix) if (outfile_prefix and cfg.run.get("save_dir")) else None,
        )
        
        categories = self.metric._coco_api.loadCats(self.metric._coco_api.getCatIds())
        class_names = [category['name'] for category in categories]
        self.metric.dataset_meta = dict(classes=class_names)
        
        self.score_threshold = cfg.model.evaluator.score_thr
        self.mask_threshold = cfg.model.evaluator.mask_thr
        self.nms_threshold = cfg.model.evaluator.nms_thr
        