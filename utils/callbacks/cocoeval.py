import os
from os.path import join
from itertools import islice
import wandb
from pytorch_lightning import Callback
from configs import cfg as _cfg
from utils.registry import CALLBACKS

from visualizations.coco_vis import save_coco_vis


@CALLBACKS.register(name="CocoEval")
class CocoEval(Callback):
    def __init__(self, save_coco_vis=False, alpha=0.65,
                 draw_border=True, border_size=15, border_color='same', 
                 static_color=False, show_img=False, save_dir=None):
        super().__init__()
        self.save_coco_vis = save_coco_vis
        self.save_dir = save_dir
        self.alpha = alpha
        self.draw_border = draw_border
        self.border_size = border_size
        self.border_color = border_color
        self.static_color = static_color
        self.show_img = show_img
        self._evaluators = None

    def setup(self, trainer, pl_module, stage=None):
        super().setup(trainer, pl_module, stage)

        if self.save_dir is None:
            if hasattr(trainer, "logger") and hasattr(trainer.logger, "save_dir"):
                self.save_dir = trainer.logger.save_dir
            else:
                self.save_dir = trainer.default_root_dir

    @property
    def evaluators(self):
        return self._evaluators
    
    @evaluators.setter
    def evaluators(self, value):
        self._evaluators = value

    def _run_process(self, outputs, phase):
        if outputs is None:
            return
        targets = outputs["targets"]
        preds = outputs["preds"]

        # TODO: this should be handled in the model
        preds["img_id"] = [targets[i]["img_id"] for i in range(len(targets))]
        preds["ori_shape"] = [targets[i]["ori_shape"] for i in range(len(targets))]
        preds["resized_shape"] = [targets[i]["resized_shape"] for i in range(len(targets))]

        for name, evaluator in self.evaluators[phase].items():
            evaluator.process(preds)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._run_process(outputs, phase="valid")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._run_process(outputs, phase="test")


    def _run_eval(self, evaluators, pl_module, dataloader, phase):
        epoch = pl_module.current_epoch if hasattr(pl_module, 'current_epoch') else None

        for name, evaluator in evaluators.items():
            print(f"Evaluating {name}...")
            evaluator.model = pl_module
            evaluator.evaluate(verbose=True)
            stats = evaluator.stats

            # Log metrics
            for key, val in stats.items():
                pl_module.log(f"metrics/{phase}/{key}", val, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            if wandb.run is not None:
                for key, val in stats.items():
                    wandb.log({f"metrics/{phase}/{key}": val}, step=pl_module.global_step)

            # Save COCO visualizations
            if self.save_coco_vis and "coco" in name:
                save_dir = join(self.save_dir, f"{phase}_visuals", f"epoch_{epoch}", "results")
                os.makedirs(save_dir, exist_ok=True)
                
                gt_coco = evaluator.gt_coco
                pred_coco = evaluator.pred_coco

                for batch in islice(dataloader, 1):
                    targets = batch[0]
                    img = targets["image"][0]
                    fname = targets["file_name"]
                    idx = targets["coco_id"]
                    H, W = targets["ori_shape"]

                    vis_path = f"{save_dir}/pred_{name}_{fname}.jpg"
                    save_coco_vis(img, gt_coco, pred_coco, idx, shape=[H, W], 
                                  alpha=self.alpha, draw_border=self.draw_border, 
                                  border_size=self.border_size, border_color=self.border_color, 
                                  static_color=self.static_color, show_img=self.show_img,
                                  path=vis_path)


    def on_validation_epoch_end(self, trainer, pl_module):
        val_loader = trainer.val_dataloaders
        if isinstance(val_loader, (list, tuple)):
            val_loader = val_loader[0]
        self._run_eval(self.evaluators["valid"], pl_module, val_loader, phase="valid")

    def on_test_epoch_end(self, trainer, pl_module):
        test_loader = trainer.test_dataloaders
        if isinstance(test_loader, (list, tuple)):
            test_loader = test_loader[0]
        self._run_eval(self.evaluators["test"], pl_module, test_loader, phase="test")
