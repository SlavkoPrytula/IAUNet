import time
import datetime
import torch

from utils.callbacks import Callback
from utils.registry import CALLBACKS

from utils.rank_zero import rank_zero_only


@CALLBACKS.register(name="LossLoggerCallback")
class LossLoggerCallback(Callback):
    def __init__(self, log_every_n_steps=10):
        self.log_every_n_steps = log_every_n_steps
        self.start_time = None

    def on_train_epoch_start(self, trainer, cfg, epoch, **kwargs):
        self.start_time = time.time()

    def on_valid_epoch_start(self, trainer, cfg, epoch, **kwargs):
        self.start_time = time.time()
    
    def on_train_batch_end(self, trainer, cfg, batch, **kwargs):
        self.log_step(trainer, cfg, batch, stage="train")

    def on_valid_batch_end(self, trainer, cfg, batch, **kwargs):
        self.log_step(trainer, cfg, batch, stage="valid")

    def on_train_epoch_end(self, trainer, cfg, epoch, **kwargs):
        self.log_epoch(trainer, cfg, epoch, stage="train")

    def on_valid_epoch_end(self, trainer, cfg, epoch, **kwargs):
        self.log_epoch(trainer, cfg, epoch, stage="valid")
        

    @rank_zero_only
    def log_step(self, trainer, cfg, batch, stage="train"):
        if batch % self.log_every_n_steps == 0:

            optimizer = trainer.optimizer
            total_steps = trainer.train_loop.total_steps if stage == "train" else trainer.valid_loop.total_steps
            logger = trainer.logger
            epoch = trainer.current_epoch
            loss = trainer.loss
        
            mem = torch.cuda.memory_reserved() / 1E6 if torch.cuda.is_available() else 0
            current_lr = optimizer.param_groups[0]['lr']
            
            elapsed_time = time.time() - self.start_time
            iters_done = batch + 1
            iters_left = total_steps - iters_done
            avg_iter_time = elapsed_time / iters_done
            eta = avg_iter_time * iters_left
            eta = str(datetime.timedelta(seconds=int(eta)))

            logger.info(f'Epoch({stage}) [{epoch}][{batch}/{total_steps}] loss: {loss:.4f}, eta: {eta}, lr: {current_lr:.6f}, mem: {mem:.0f}')


    @rank_zero_only
    def log_epoch(self, trainer, cfg, epoch, stage="train"):
        logger = trainer.logger
        loss_dict = trainer.loss_dict
        
        print()
        for l in loss_dict:
            logger.info(f'{l}: {loss_dict[l]:.4f}')
        print()