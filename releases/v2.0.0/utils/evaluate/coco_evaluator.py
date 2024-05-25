import torch
from torch import nn

from utils.coco.coco import COCO
from utils.coco.cocoeval import COCOeval
# from pycocotools.cocoeval import COCOeval

# coco_eval
class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()

        self.gt_coco = {}
        self.pred_coco = {}

        # inference
        self.score_threshold = cfg.score_thr
        self.mask_threshold = cfg.mask_thr

        self.coco_eval = None
        self.stats = {
            "mAP@0.5:0.95": -1, 
            "mAP@0.5": -1, 
            "mAP@0.75": -1,
            "mAP(s)@0.5": -1,
            "mAP(m)@0.5": -1,
            "mAP(l)@0.5": -1,
            }


    @torch.no_grad()
    def inference_single(self, model, input):
        model.eval()
        output = model(input)
        return output


    def evaluate(self, verbose=False):
        # Create COCO evaluation object for segmentation
        gt_coco = COCO(self.gt_coco, verbose=verbose)
        pred_coco = COCO(self.pred_coco, verbose=verbose)
        self.coco_eval = COCOeval(gt_coco, pred_coco, iouType='segm', verbose=verbose)
        # self.coco_eval = COCOeval(gt_coco, pred_coco, iouType='segm')

        # Run the evaluation
        self.coco_eval.evaluate()
        self.coco_eval.accumulate()
        self.coco_eval.summarize()

        stats = self.coco_eval.stats
        for index, key in enumerate(self.stats):
            if index < len(stats):
                self.stats[key] = stats[index]
