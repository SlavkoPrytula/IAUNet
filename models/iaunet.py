import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from configs import cfg as _cfg
import torch.nn.functional as F

from utils.registry import MODELS, DECODERS, OPTIMIZERS, SCHEDULERS, CRITERIONS
from utils.utils import nested_tensor_from_tensor_list
from configs import cfg as _cfg
import wandb


@MODELS.register(name="iaunet")
class IAUNet(pl.LightningModule):
    def __init__(self, cfg: _cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder = MODELS.build(cfg.model.encoder)
        embed_dims = self.encoder.embed_dims
        self.decoder = DECODERS.build(cfg.model.decoder, embed_dims=embed_dims)
        self.criterion = self.configure_criterion()

        self.register_buffer("pixel_mean", torch.tensor(cfg.dataset.mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.dataset.std).view(-1, 1, 1), False)
        self.size_divisibility = 32

    def forward(self, batch):
        x = batch["images"]
        tgt = batch["targets"]
        
        max_shape = x.shape[-2:]
        skips = self.encoder(x)
        out = self.decoder(skips, max_shape)
        return out
    
    def prepare_batch(self, batch):
        """Utility to extract images and process targets."""
        images = []
        targets = []

        for sample in batch:
            ignore = ["img_id", "img_path", "ori_shape", "file_name", "coco_id", "resized_shape"]
            sample = {k: v.to(self.device) if k not in ignore else v for k, v in sample.items()}
            images.append(sample["image"])
            targets.append(sample)

        # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # images = nested_tensor_from_tensor_list(images)

        # # pad images and segmentations here.
        # if self.size_divisibility > 1:
        #     image_size = images.tensors.shape[-2:]
        #     pad_h = (self.size_divisibility - image_size[0] % self.size_divisibility) % self.size_divisibility
        #     pad_w = (self.size_divisibility - image_size[1] % self.size_divisibility) % self.size_divisibility
        #     padding_size = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
        #     # pad image
        #     images.tensors = F.pad(images.tensors, padding_size, value=0)

        # # prepare targets.
        # targets = self.prepare_targets(targets, images)

        images = nested_tensor_from_tensor_list(images)

        return images, targets
    
    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensors.shape[-2:]
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.get("instance_masks")
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, :gt_masks.shape[1], :gt_masks.shape[2]] = gt_masks
            targets_per_image["instance_masks"] = padded_masks
        return targets

    def _shared_step(self, batch):
        """Shared step logic for forward pass and loss computation."""
        images, targets = self.prepare_batch(batch)
        batch_data = {"images": images.tensors, "targets": targets}
        preds = self(batch_data)
        loss_dict, _ = self.criterion(preds, targets, return_matches=True, epoch=self.current_epoch)
        total_loss = sum(loss_dict.values())
        
        return total_loss, loss_dict, preds, targets, images.tensors.size(0)


    def training_step(self, batch, batch_idx):
        """Training step - returns loss for backpropagation."""
        total_loss, loss_dict, preds, targets, batch_size = self._shared_step(batch)
        
        self.log("train/loss_total", total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, 
                    on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        
        if wandb.run:
            wandb.log({
                "train/loss_total": total_loss.item(),
                **{f"train/{k}": v.item() for k, v in loss_dict.items()},
            }, step=self.global_step)
        
        return total_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """Validation step - no gradients needed."""
        total_loss, loss_dict, preds, targets, batch_size = self._shared_step(batch)
        
        self.log("valid/loss_total", total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        for k, v in loss_dict.items():
            self.log(f"valid/{k}", v, 
                    on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
            
        if wandb.run:
            wandb.log({
                "valid/loss_total": total_loss.item(),
                **{f"valid/{k}": v.item() for k, v in loss_dict.items()},
            }, step=self.global_step)
        
        return {"loss": total_loss, "preds": preds, "targets": targets}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        """Test step - no gradients needed."""
        total_loss, loss_dict, preds, targets, batch_size = self._shared_step(batch)
        
        self.log("test/loss_total", total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        for k, v in loss_dict.items():
            self.log(f"test/{k}", v, 
                    on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)

        if wandb.run:
            wandb.log({
                "test/loss_total": total_loss.item(),
                **{f"test/{k}": v.item() for k, v in loss_dict.items()},
            }, step=self.global_step)
        
        return {"loss": total_loss, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        _optimizer = self.configure_optimizer()
        _scheduler = self.configure_scheduler(_optimizer)

        return {
            "optimizer": _optimizer,
            "lr_scheduler": {
                "scheduler": _scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def configure_optimizer(self):
        solver_cfg = OmegaConf.to_container(self.cfg.model.solver, resolve=True)
        optimizer_cfg = solver_cfg['optimizer']
        optimizer_cfg['params'] = self.parameters()
        return OPTIMIZERS.build(optimizer_cfg)

    def configure_scheduler(self, optimizer):
        max_steps = self.trainer.estimated_stepping_batches
        scheduler_cfg = OmegaConf.to_container(self.cfg.model.scheduler, resolve=True)
        scheduler_cfg["T_max"] = max_steps
        scheduler_cfg["optimizer"] = optimizer
        return SCHEDULERS.build(scheduler_cfg)

    def configure_criterion(self):
        self.cfg.model.criterion.num_classes = self.cfg.model.decoder.num_classes
        return CRITERIONS.build(self.cfg.model.criterion)
