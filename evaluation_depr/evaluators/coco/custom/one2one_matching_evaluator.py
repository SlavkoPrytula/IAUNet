import torch
from os.path import join
from configs import cfg
from tqdm import tqdm

from ..mmdet_dataloader_evaluator import MMDetDataloaderEvaluator

from utils.utils import nested_tensor_from_tensor_list
from utils.opt.mask_nms import mask_nms
from utils.registry import EVALUATORS, DATASETS, build_criterion
from evaluation.mmdet import CocoMetric

from utils.common.decorators import timeit_evaluator, memory_evaluator


@timeit_evaluator
@memory_evaluator
@EVALUATORS.register(name="One2OneMatchingEvaluator")
class One2OneMatchingEvaluator(MMDetDataloaderEvaluator):
    def __init__(self, cfg: cfg, model=None, dataset=None, **kwargs):
        super().__init__(cfg, model, dataset, **kwargs)
        self.criterion = build_criterion(cfg=cfg.model.criterion)

    @torch.no_grad()
    def inference_single(self, input, targets):
        preds = self.model(input)
        _, (src_idx, tgt_idx) = self.criterion(preds, targets, self.cfg.valid.size, 
                                               return_matches=True, epoch=None)
        
        # match gt and preds.
        preds['pred_masks'] = preds['pred_masks'][src_idx].unsqueeze(0)
        preds['pred_logits'] = preds['pred_logits'][src_idx].unsqueeze(0)
        preds['pred_scores'] = preds['pred_scores'][src_idx].unsqueeze(0)
        preds['pred_bboxes'] = preds['pred_bboxes'][src_idx].unsqueeze(0)
    
        return preds


    def forward(self, dataloader):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), miniters=5)
        for step, batch in pbar:
            if batch is None:
                continue
            
            # prepare targets
            images = []
            targets = []

            target = batch[0]
            ignore = ["img_id", "img_path", "ori_shape", "file_name", "coco_id"]
            target = {k: v.to(cfg.device) if k not in ignore else v 
                    for k, v in target.items()}
            images.append(target["image"])
            targets.append(target)

            image = nested_tensor_from_tensor_list(images)


            # ============= PREDICTION ==============
            # predict.
            pred = self.inference_single(image.tensors, targets)

            pred["img_id"] = target["img_id"]
            pred["ori_shape"] = target["ori_shape"]

            self.process(pred)

    