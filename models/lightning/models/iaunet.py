import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from utils.registry import MODELS, DECODERS, OPTIMIZERS, SCHEDULERS, CRITERIONS
from utils.utils import nested_tensor_from_tensor_list
from configs import cfg as _cfg


@MODELS.register(name="iaunet")
class IAUNet(pl.LightningModule):
    def __init__(self, cfg: _cfg, evaluators):
        super().__init__()
        self.cfg = cfg

        self.encoder = MODELS.build(cfg.model.encoder)
        embed_dims = self.encoder.embed_dims
        self.decoder = DECODERS.build(cfg.model.decoder, embed_dims=embed_dims)
        self.criterion = self.configure_criterion()
        self.evaluators = evaluators
        self.results = {}


    def forward(self, x):
        max_shape = x.shape[-2:]
        skips = self.encoder(x)
        return self.decoder(skips, max_shape)
    
    def _prepare_batch(self, batch):
        """Utility to extract images and process targets."""
        images = []
        targets = []

        for sample in batch:
            ignore = ["img_id", "img_path", "ori_shape", "file_name", "coco_id"]
            sample = {k: v.to(self.device) if k not in ignore else v for k, v in sample.items()}
            images.append(sample["image"])
            targets.append(sample)

        images = nested_tensor_from_tensor_list(images)
        return images, targets


    def training_step(self, batch, batch_idx):
        """Training step."""
        images, targets = self._prepare_batch(batch)
        preds = self(images.tensors)
        loss_dict, _ = self.criterion(preds, targets, return_matches=True, epoch=self.current_epoch)
        loss = sum(loss_dict.values())
        batch_size = images.tensors.size(0)

        self.log("train/loss_total", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss


    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, targets = self._prepare_batch(batch)
        preds = self(images.tensors)
        loss_dict, _ = self.criterion(preds, targets, return_matches=True, epoch=self.current_epoch)
        loss = sum(loss_dict.values())
        batch_size = images.tensors.size(0)

        preds["img_id"] = [targets[i]["img_id"] for i in range(len(targets))]
        preds["ori_shape"] = [targets[i]["ori_shape"] for i in range(len(targets))]

        for name, evaluator in self.evaluators["valid"].items():
            evaluator.process(preds)

        self.log("val/loss_total", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return {"loss": loss, "preds": preds, "targets": targets}

    
    def on_train_epoch_end(self):
        # get epoch loss.
        epoch_loss = self.trainer.callback_metrics.get("train/loss_total", None)
        self.results["loss_train"] = epoch_loss


    def on_validation_epoch_end(self):
        # get epoch loss.
        epoch_loss = self.trainer.callback_metrics.get("val/loss_total", None)
        self.results["loss_valid"] = epoch_loss

        # evaluate.
        for name, evaluator in self.evaluators["valid"].items():
            evaluator.evaluate(verbose=True)
            stats = evaluator.stats
            self.results.update(stats)

            # Log metrics
            for key, val in stats.items():
                self.log(f"metrics/valid/{key}", val)

        metrics = ['epoch'] + list(self.results.keys())
        vals = [self.current_epoch] + list(self.results.values())
        self._save_results_csv(metrics, vals)


    def _save_results_csv(self, metrics, vals):
        csv_path = self.cfg.run.save_dir / 'results.csv'
        s = '' if csv_path.exists() else (('%13s,' * (len(metrics)) % tuple(metrics)).rstrip(',') + '\n')  # header
        with open(csv_path, 'a') as f:
            f.write(s + ('%13.5g,' * (len(metrics)) % tuple(vals)).rstrip(',') + '\n')


    def test_step(self, batch, batch_idx):
        images, targets = self._prepare_batch(batch)
        preds = self(images.tensors)
        loss_dict, _ = self.criterion(preds, targets, return_matches=True, epoch=self.current_epoch)
        loss = sum(loss_dict.values())
        batch_size = images.tensors.size(0)

        self.log("test/loss_total", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        for k, v in loss_dict.items():
            self.log(f"test/{k}", v, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        return {"loss": loss, "preds": preds, "targets": targets}


    def configure_optimizers(self):
        _optimizer = self.configure_optimizer()
        _scheduler = self.configure_scheduler(_optimizer)

        return {
            "optimizer": _optimizer,
            "lr_scheduler": {
                "scheduler": _scheduler,
            },
        }

    def configure_optimizer(self):
        solver_cfg = OmegaConf.to_container(self.cfg.model.solver, resolve=True)
        optimizer_cfg = solver_cfg['optimizer']
        optimizer_cfg['params'] = self.parameters()
        return OPTIMIZERS.build(optimizer_cfg)

    def configure_scheduler(self, optimizer):
        scheduler_cfg = OmegaConf.to_container(self.cfg.model.scheduler, resolve=True)
        scheduler_cfg["optimizer"] = optimizer
        return SCHEDULERS.build(scheduler_cfg)

    def configure_criterion(self):
        self.cfg.model.criterion.num_classes = self.cfg.model.decoder.num_classes
        # criterion_cfg = OmegaConf.to_container(self.cfg.model.criterion, resolve=True)
        return CRITERIONS.build(self.cfg.model.criterion)


