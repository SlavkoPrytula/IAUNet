import time
import datetime
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class LossLoggerCallback(Callback):
    def __init__(self, log_every_n_steps=10):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.start_time = None
        self.last_losses = {}

    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_validation_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Try to get the latest loss dict from outputs or pl_module
        loss_dict = outputs.get('loss_dict') if isinstance(outputs, dict) and 'loss_dict' in outputs else getattr(pl_module, 'loss_dict', {})
        self.last_losses = loss_dict
        self.log_step(trainer, pl_module, batch_idx, stage="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss_dict = outputs.get('loss_dict') if isinstance(outputs, dict) and 'loss_dict' in outputs else getattr(pl_module, 'loss_dict', {})
        self.last_losses = loss_dict
        self.log_step(trainer, pl_module, batch_idx, stage="valid")

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_epoch(trainer, pl_module, stage="train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_epoch(trainer, pl_module, stage="valid")

    @rank_zero_only
    def log_step(self, trainer, pl_module, batch_idx, stage="train"):
        if batch_idx % self.log_every_n_steps != 0:
            return
        current_epoch = trainer.current_epoch

        # Use logged_metrics for total loss, fallback to 0.0
        loss = trainer.logged_metrics.get(f"{stage}/loss_total", 0.0)
        mem = torch.cuda.memory_reserved() / 1e6 if torch.cuda.is_available() else 0
        lr = trainer.optimizers[0].param_groups[0]['lr'] if trainer.optimizers else 0.0

        # Estimate time
        elapsed_time = time.time() - self.start_time
        iters_done = batch_idx + 1

        # Fix: handle DataLoader or list/tuple of DataLoaders
        if stage == "train":
            total_steps = len(trainer.train_dataloader)
        else:
            val_loader = trainer.val_dataloaders
            if isinstance(val_loader, (list, tuple)):
                val_loader = val_loader[0]
            total_steps = len(val_loader)
        iters_left = total_steps - iters_done
        avg_iter_time = elapsed_time / iters_done if iters_done > 0 else 0
        eta = str(datetime.timedelta(seconds=int(avg_iter_time * iters_left))) if avg_iter_time > 0 else "0:00:00"

        # Log metrics
        trainer.logger.log_metrics({f"{stage}/eta": eta, f"{stage}/lr": lr, f"{stage}/mem": mem}, step=trainer.global_step)
        
        print(f"[{stage.upper()}][Epoch {current_epoch}][{batch_idx}/{total_steps}] "
              f"eta: {eta}, lr: {lr:.6f}, mem: {mem:.0f}MB")

    @rank_zero_only
    def log_epoch(self, trainer, pl_module, stage="train"):
        if not self.last_losses:
            return
        print()
        for k, v in self.last_losses.items():
            trainer.logger.log_metrics({f"{stage}/{k}": v}, step=trainer.global_step)
            print(f"{stage}/{k}: {v:.4f}")
        print()
