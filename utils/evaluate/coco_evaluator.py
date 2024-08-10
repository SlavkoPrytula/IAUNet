import torch
from torch import nn

from utils.coco.coco import COCO
from utils.coco.cocoeval import COCOeval
# from pycocotools.cocoeval import COCOeval
from configs import cfg

# coco_eval
class Evaluator(nn.Module):
    def __init__(self, cfg: cfg, model=None, **kwargs):
        super(Evaluator, self).__init__()
        self.cfg = cfg
        self.gt_coco = {}
        self.pred_coco = {}
        self.model = model
        self.device = next(model.parameters()).device
        if model:
            self.model.eval()
        else:
            print("WARNING: model is None, model should be correctly passed to the Evaluator")

        # inference
        # self.score_threshold = cfg.score_thr
        # self.mask_threshold = cfg.mask_thr
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

    def forward(self, *args, **kwargs):
        if not callable(getattr(self.model, "inference", None)):
            print("UserWarning: In the new release v2.1.0 model classes should have inference methods!")
        pass

    def process(self, pred: dict):
        ...


    @torch.no_grad()
    def inference_single(self, input):
        output = self.model(input)
        return output


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
