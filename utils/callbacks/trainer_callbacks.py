import time
import datetime
import torch


class LossLoggerCallback:
    def __init__(self, logger, optimizer, total_steps, log_interval=10):
        self.logger = logger
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.start_time = time.time()


    def on_train_batch_end(self, step, epoch, loss):
        self.log_loss(step, epoch, loss, stage="train")


    def on_valid_batch_end(self, step, epoch, loss):
        self.log_loss(step, epoch, loss, stage="valid")


    def log_loss(self, step, epoch, loss, stage="train"):
        if step % self.log_interval == 0:
            mem = torch.cuda.memory_reserved() / 1E6 if torch.cuda.is_available() else 0
            current_lr = self.optimizer.param_groups[0]['lr']
            
            elapsed_time = time.time() - self.start_time
            iters_done = step + 1
            iters_left = self.total_steps - iters_done
            avg_iter_time = elapsed_time / iters_done
            eta = avg_iter_time * iters_left
            eta = str(datetime.timedelta(seconds=int(eta)))

            self.logger.info(f'Epoch({stage}) [{epoch}][{step}/{self.total_steps}] loss: {loss:.4f}, eta: {eta}, lr: {current_lr:.6f}, mem: {mem:.0f}')
