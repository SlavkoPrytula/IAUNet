import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from configs import cfg as _cfg

from utils.registry import MODELS, DECODERS, OPTIMIZERS, SCHEDULERS, CRITERIONS
from utils.utils import nested_tensor_from_tensor_list
from configs import cfg as _cfg


@MODELS.register(name="iaunet")
class IAUNet(pl.LightningModule):
    def __init__(self, cfg: _cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder = MODELS.build(cfg.model.encoder)
        embed_dims = self.encoder.embed_dims
        self.decoder = DECODERS.build(cfg.model.decoder, embed_dims=embed_dims)
        self.criterion = self.configure_criterion()
        self.results = {}

        # self.register_buffer("pixel_mean", torch.Tensor(cfg.dataset.mean).view(-1, 1, 1), False)
        # self.register_buffer("pixel_std", torch.Tensor(cfg.dataset.std).view(-1, 1, 1), False)

    def forward(self, batch):
        x = batch["images"]
        tgt = batch["targets"]
        
        max_shape = x.shape[-2:] # (img_h, img_w)
        skips = self.encoder(x)
        out = self.decoder(skips, max_shape) 
        return out
    
    def _prepare_batch(self, batch):
        """Utility to extract images and process targets."""
        images = []
        targets = []

        for sample in batch:
            ignore = ["img_id", "img_path", "ori_shape", "file_name", "coco_id", "resized_shape"]
            sample = {k: v.to(self.device) if k not in ignore else v for k, v in sample.items()}
            images.append(sample["image"])
            targets.append(sample)

        images = nested_tensor_from_tensor_list(images)
        return images, targets

    def _shared_step(self, batch, batch_idx, prefix):
        images, targets = self._prepare_batch(batch)
        batch = {"images": images.tensors, "targets": targets}
        preds = self(batch)
        loss_dict, _ = self.criterion(preds, targets, return_matches=True, epoch=self.current_epoch)
        loss = sum(loss_dict.values())
        batch_size = images.tensors.size(0)

        self.log(f"{prefix}/loss_total", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        for k, v in loss_dict.items():
            self.log(f"{prefix}/{k}", v, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return {"loss": loss, "preds": preds, "targets": targets}


    def training_step(self, batch, batch_idx):
        """Training step."""
        return self._shared_step(batch, batch_idx, prefix="train")

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self._shared_step(batch, batch_idx, prefix="valid")

    def test_step(self, batch, batch_idx):
        """Test step."""
        return self._shared_step(batch, batch_idx, prefix="test")

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
        return CRITERIONS.build(self.cfg.model.criterion)

