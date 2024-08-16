from utils.coco.coco import COCO
from utils.coco.cocoeval import COCOeval
from configs import cfg

from ..base_evaluator import BaseEvaluator


# coco_eval
class COCOEvaluator(BaseEvaluator):
    def __init__(self, cfg: cfg, model=None, **kwargs):
        super().__init__(cfg, model, **kwargs)
        self.gt_coco = {}
        self.pred_coco = {}

        # inference
        self.score_threshold = cfg.model.evaluator.score_thr
        self.mask_threshold = cfg.model.evaluator.mask_thr

        self.coco_eval = None
        self.stats = {
            "mAP@0.5:0.95": 0, 
            "mAP@0.5": 0, 
            "mAP@0.75": 0,
            "mAP(s)@0.5": 0,
            "mAP(m)@0.5": 0,
            "mAP(l)@0.5": 0,
            }

    def evaluate(self, verbose=False):
        # Create COCO evaluation object for segmentation
        self.gt_coco = COCO(self.gt_coco, verbose=verbose)
        self.pred_coco = COCO(self.pred_coco, verbose=verbose)
        self.coco_eval = COCOeval(self.gt_coco, self.pred_coco, iouType='segm', verbose=verbose)

        # Run the evaluation
        self.coco_eval.evaluate()
        self.coco_eval.accumulate()
        self.coco_eval.summarize()

        stats = self.coco_eval.stats
        for index, key in enumerate(self.stats):
            if index < len(stats):
                self.stats[key] = stats[index]
