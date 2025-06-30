import torch
import time
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from utils.registry import CALLBACKS
from collections import OrderedDict


@CALLBACKS.register(name="ProgressBar")
class ProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate=10):
        super().__init__(refresh_rate=refresh_rate)
        self.start_time = None
        self.total_steps = None

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        self.start_time = time.time()
        
        current_epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs
        self.train_progress_bar.set_description(f"Epoch {current_epoch}/{max_epochs}")

    def get_metrics(self, trainer, pl_module):
        super().get_metrics(trainer, pl_module)
        ordered = OrderedDict()
        
        # memory.
        mem = torch.cuda.memory_reserved() / 1e6 if torch.cuda.is_available() else 0
        ordered['mem(MB)'] = f"{mem:.0f}"
        
        # eta.
        eta_str = ""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            steps_done = trainer.global_step
            if self.total_steps is None:
                self.total_steps = trainer.estimated_stepping_batches
            total_steps = self.total_steps
            if steps_done > 0:
                eta = (elapsed / steps_done) * (total_steps - steps_done)
                h = int(eta // 3600)
                m = int((eta % 3600) // 60) 
                s = int(eta % 60)
                eta_str = f"{h:02d}:{m:02d}:{s:02d}"
                ordered['eta'] = eta_str

        # learning rate.
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0]['lr']
            ordered['lr'] = f"{lr:.6f}"
        
        # losses.
        for k, v in trainer.progress_bar_metrics.items():
            if 'loss' in k:
                ordered[k] = f"{v:.4f}"

        return ordered