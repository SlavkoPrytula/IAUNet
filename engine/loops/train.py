import wandb
from typing import Dict

from torch.cuda import amp
from utils.utils import nested_tensor_from_tensor_list

from evaluation.evaluators import BaseEvaluator
from configs import cfg
from .base import BaseLoop


class TrainLoop(BaseLoop):
    def __init__(
        self,
        cfg: cfg, 
        model, 
        criterion,
        optimizer, 
        scheduler, 
        dataloader, 
        device, 
        callbacks,
        logger,
        evaluators: Dict[str, BaseEvaluator], 
    ):
        super().__init__(
            cfg, 
            model, 
            criterion, 
            dataloader, 
            device, 
            logger, 
            callbacks,
            evaluators
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = amp.GradScaler()


    def run(self):
        self.model.train()
        results = {}
        running_loss = 0.0
        dataset_size = 0

        self.trigger_callbacks('on_train_epoch_start', trainer=self.trainer, cfg=self.cfg, epoch=self.epoch)

        self.logger.info('Loss/Train')
        for step, batch in enumerate(self.dataloader):
            if batch is None:
                continue

            # prepare targets
            images = []
            targets = []
            for i in range(len(batch)):
                target = batch[i]
                ignore = ["img_id", "img_path", "ori_shape", "file_name", "coco_id"]
                target = {k: v.to(self.device) if k not in ignore else v 
                        for k, v in target.items()}
                images.append(target["image"])
                targets.append(target)
                
            images = nested_tensor_from_tensor_list(images)   # (B, C, H, W)
            batch_size = images.tensors.size(0)
            
            with amp.autocast(enabled=True):
                output = self.model(images.tensors) # (B, N, H, W)
                
                # get losses
                loss_dict, (src_idx, tgt_idx) = self.criterion(output, targets, 
                                                                [self.cfg.dataset.train_dataset.size], 
                                                                return_matches=True, epoch=self.epoch)
                loss = sum(loss_dict.values())
                
            self.scaler.scale(loss).backward()
            if (step + 1) % 1 == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()
                    
            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size
            epoch_loss = running_loss / dataset_size
            self.trainer.loss = epoch_loss

            self.trigger_callbacks('on_train_batch_end', trainer=self.trainer, cfg=self.cfg, batch=step)

        # wandb results.
        # TODO: check from cfg
        if wandb.run is not None:
            wandb.log({f"train/loss_train": epoch_loss})
            for l in loss_dict:
                wandb.log({f"train/{l}_train": loss_dict[l]})

        # logging results.      
        results["loss_train"] = epoch_loss
        for l in loss_dict:
            results[f"{l}_train"] = loss_dict[l]

        self.trainer.output = output
        self.trainer.loss_dict = loss_dict
        self.trigger_callbacks('on_train_epoch_end', trainer=self.trainer, cfg=self.cfg, epoch=self.epoch)
        
        return results


# import time
# import numpy as np
# from tqdm import tqdm
# class TrainLoop(BaseLoop):
#     def __init__(
#         self,
#         cfg: cfg, 
#         model, 
#         criterion,
#         optimizer, 
#         scheduler, 
#         dataloader, 
#         device, 
#         callbacks,
#         logger,
#         evaluators, 
#     ):
#         super().__init__(
#             cfg, 
#             model, 
#             criterion, 
#             dataloader, 
#             device, 
#             logger, 
#             callbacks,
#             evaluators
#         )
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.scaler = amp.GradScaler()
#         self.total_steps = len(self.dataloader)

#     def run(self):
#         epoch_start_time = time.time()  # Start time for the entire epoch

#         self.model.train()
#         results = {}
#         running_loss = 0.0
#         dataset_size = 0

#         # Timing storage
#         batch_times = []
#         forward_times = []
#         loss_times = []
#         backward_times = []
#         total_times = []

#         self.trigger_callbacks('on_train_epoch_start', trainer=self.trainer, cfg=self.cfg, epoch=self.epoch)

#         self.logger.info('Loss/Train')
#         for step, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
#             if batch is None:
#                 continue
            
#             # Start timing for the entire batch
#             total_start_time = time.time()
#             batch_start_time = time.time()

#             # prepare targets
#             images = []
#             targets = []
#             for i in range(len(batch)):
#                 target = batch[i]
#                 ignore = ["img_id", "img_path", "ori_shape", "file_name", "coco_id"]
#                 target = {k: v.to(self.device) if k not in ignore else v 
#                         for k, v in target.items()}
#                 images.append(target["image"])
#                 targets.append(target)
                
#             images = nested_tensor_from_tensor_list(images)   # (B, C, H, W)
#             batch_size = images.tensors.size(0)
#             batch_end_time = time.time()
            
#             forward_start_time = time.time()
#             with amp.autocast(enabled=True):
#                 output = self.model(images.tensors) # (B, N, H, W)
                
#                 # get losses
#                 loss_start_time = time.time()
#                 loss_dict, (src_idx, tgt_idx) = self.criterion(output, targets, 
#                                                                 [self.cfg.dataset.train_dataset.size], 
#                                                                 return_matches=True, epoch=self.epoch)
#                 loss = sum(loss_dict.values())
#                 loss_end_time = time.time()
                
#             forward_end_time = time.time()
            
#             backward_start_time = time.time()
#             self.scaler.scale(loss).backward()
#             if (step + 1) % 1 == 0:
#                 self.scaler.step(self.optimizer)
#                 self.scaler.update()
#                 self.optimizer.zero_grad()
#                 if self.scheduler is not None:
#                     self.scheduler.step()
#             backward_end_time = time.time()
            
#             total_end_time = time.time()

#             # Collecting times
#             batch_times.append(batch_end_time - batch_start_time)
#             forward_times.append(forward_end_time - forward_start_time)
#             loss_times.append(loss_end_time - loss_start_time)
#             backward_times.append(backward_end_time - backward_start_time)
#             total_times.append(total_end_time - total_start_time)
            
#             running_loss += (loss.item() * batch_size)
#             dataset_size += batch_size
#             epoch_loss = running_loss / dataset_size
#             self.trainer.loss = epoch_loss

#             self.trigger_callbacks('on_train_batch_end', trainer=self.trainer, cfg=self.cfg, batch=step)

#         # wandb results.
#         if wandb.run is not None:
#             wandb.log({f"train/loss_train": epoch_loss})
#             for l in loss_dict:
#                 wandb.log({f"train/{l}_train": loss_dict[l]})

#         # logging results.      
#         results["loss_train"] = epoch_loss
#         for l in loss_dict:
#             results[f"{l}_train"] = loss_dict[l]

#         # Logging the average times
#         self.logger.info(f"Avg. Batch Time: {np.mean(batch_times):.4f}s")
#         self.logger.info(f"Avg. Forward Time: {np.mean(forward_times):.4f}s")
#         self.logger.info(f"Avg. Loss Time: {np.mean(loss_times):.4f}s")
#         self.logger.info(f"Avg. Backward Time: {np.mean(backward_times):.4f}s")
#         self.logger.info(f"Avg. Total Time: {np.mean(total_times):.4f}s")
        
#         # Logging the total time for the epoch
#         epoch_end_time = time.time()
#         self.logger.info(f"Epoch Time: {epoch_end_time - epoch_start_time:.4f}s")

#         self.trainer.output = output
#         self.trainer.loss_dict = loss_dict
#         self.trigger_callbacks('on_train_epoch_end', trainer=self.trainer, cfg=self.cfg, epoch=self.epoch)
        
#         return results