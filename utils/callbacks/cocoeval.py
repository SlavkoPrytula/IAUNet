import os
from os.path import join
from itertools import islice
import wandb
from pytorch_lightning import Callback
from configs import cfg as _cfg

from visualizations.coco_vis import save_coco_vis


class CocoEvalCallback(Callback):
    def __init__(self, cfg: _cfg, evaluators):
        super().__init__()
        self.cfg = cfg
        self.evaluators = evaluators
        self._validation_dataloader = None
        self._test_dataloader = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs is None:
            return
        targets = outputs["targets"]
        preds = outputs["preds"]

        preds["img_id"] = [targets[i]["img_id"] for i in range(len(targets))]
        preds["ori_shape"] = [targets[i]["ori_shape"] for i in range(len(targets))]

        for name, evaluator in self.evaluators["valid"].items():
            evaluator.process(preds)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs is None:
            return
        targets = outputs["targets"]
        preds = outputs["preds"]

        preds["img_id"] = [targets[i]["img_id"] for i in range(len(targets))]
        preds["ori_shape"] = [targets[i]["ori_shape"] for i in range(len(targets))]

        for name, evaluator in self.evaluators["eval"].items():
            evaluator.process(preds)


    def _run_eval(self, evaluators, pl_module, dataloader, phase):
        epoch = pl_module.current_epoch if hasattr(pl_module, 'current_epoch') else None
        save_dir = join(self.cfg.run.save_dir, f"{phase}_visuals", f"epoch_{epoch}", "results")
        os.makedirs(save_dir, exist_ok=True)

        for name, evaluator in evaluators.items():
            print(f"Evaluating {name}...")
            evaluator.model = pl_module
            evaluator.evaluate(verbose=True)
            stats = evaluator.stats

            # Log metrics
            for key, val in stats.items():
                if wandb.run:
                    wandb.log({f"metrics/{phase}/{key}": val})

            # Save COCO visualizations
            if "coco" in name:
                gt_coco = evaluator.gt_coco
                pred_coco = evaluator.pred_coco

                for batch in islice(dataloader, 2):
                    targets = batch[0]
                    img = targets["image"][0]
                    fname = targets["file_name"]
                    idx = targets["coco_id"]
                    H, W = targets["ori_shape"]

                    vis_path = f"{save_dir}/pred_{name}_{fname}.jpg"
                    save_coco_vis(img, gt_coco, pred_coco, idx, shape=[H, W], path=vis_path)

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loader = trainer.val_dataloaders
        if isinstance(val_loader, (list, tuple)):
            val_loader = val_loader[0]
        self._run_eval(self.evaluators["valid"], pl_module, val_loader, phase="valid")

    def on_test_epoch_end(self, trainer, pl_module):
        test_loader = trainer.test_dataloaders
        if isinstance(test_loader, (list, tuple)):
            test_loader = test_loader[0]
        self._run_eval(self.evaluators["eval"], pl_module, test_loader, phase="eval")
