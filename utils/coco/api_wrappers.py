import warnings
from collections import defaultdict
from pycocotools.coco import COCO as StandardCOCO
from pycocotools.cocoeval import COCOeval as StandardCOCOeval

try:
    from utils.coco.boundary_iou.coco_instance_api.coco import COCO as BoundaryCOCO
    from utils.coco.boundary_iou.coco_instance_api.cocoeval import COCOeval as BoundaryCOCOeval
except ImportError:
    BoundaryCOCO = StandardCOCO
    BoundaryCOCOeval = StandardCOCOeval
    warnings.warn("Boundary IoU API not found. Boundary IoU evaluation will not be available.", UserWarning)


class COCO(BoundaryCOCO):
    def __init__(self, annotation_file=None):
        super().__init__(annotation_file)
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)

class COCOeval:
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        self.iouType = iouType
        self.cocoGt = cocoGt
        self.cocoDt = cocoDt

        # the results are the same, but just to be sure we will separate boundary mAP from segm mAP
        if iouType == "boundary" and BoundaryCOCOeval:
            coco_eval = BoundaryCOCOeval
        else:
            coco_eval = StandardCOCOeval
        
        self.coco_eval = coco_eval(cocoGt, cocoDt, iouType=iouType)

    def evaluate(self):
        return self.coco_eval.evaluate()

    def accumulate(self):
        return self.coco_eval.accumulate()

    def summarize(self):
        return self.coco_eval.summarize()

    def __getattr__(self, name):
        return getattr(self.coco_eval, name)
