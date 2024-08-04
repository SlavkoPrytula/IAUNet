import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.rank_zero import rank_zero_only
from configs import cfg

from .base import BaseTrainer
from .loops import TrainLoop, ValidLoop


class Trainer(BaseTrainer):
    def __init__(
            self, 
            cfg: cfg, 
            model, 
            criterion, 
            optimizer, 
            scheduler, 
            train_dataloader, 
            valid_dataloader, 
            evaluators, 
            callbacks, 
            logger, 
            rank, 
            strategy
    ) -> None:
        super().__init__(
            cfg, 
            model, 
            criterion, 
            optimizer, 
            scheduler, 
            train_dataloader, 
            valid_dataloader, 
            evaluators, 
            callbacks, 
            logger, 
            rank, 
            strategy
        )

        self.train_loop = TrainLoop(
            cfg=cfg,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_dataloader,
            device=self.device,
            logger=logger,
            evaluators=evaluators,
            callbacks=callbacks,
        )
        self.train_loop.trainer = self

        self.valid_loop = ValidLoop(
            cfg=cfg,
            model=model,
            criterion=criterion,
            dataloader=valid_dataloader,
            device=self.device,
            logger=logger,
            evaluators=evaluators,
            callbacks=callbacks,
        )
        self.valid_loop.trainer = self

        self.model.to(self.device)
        if self.strategy == 'ddp':
            self.model = DDP(self.model, device_ids=[self.rank])


    def train(self):
        start = time.time()

        for epoch in range(self.max_epochs):
            self.logger.info(f'Epoch {epoch}/{self.max_epochs}')
            self.current_epoch = epoch

            results_train = self.train_loop.run()
            
            if epoch % self.check_val_every_n_epoch == 0:
                results_valid = self.valid_loop.run()

                self._update_results(results_train, results_valid)


        # saving last checkpoint.
        if self.cfg.model.save_checkpoint:
            self.save_checkpoint(self.cfg.run.save_dir / 'checkpoints/last.pth')

        end = time.time()
        time_elapsed = end - start
        self.logger.info('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))


    def _update_results(self, train_results, valid_results):
        results = {}
        results.update(valid_results)
        results.update(train_results)

        val_loss = results["loss_valid"]
        if val_loss <= self.best_loss:
            self.logger.info(f"Valid Loss Improved ({self.best_loss:.4f} ---> {val_loss:.4f})")
            self.best_loss = val_loss
            
            # saving best model.
            if self.cfg.model.save_checkpoint:
                self.save_checkpoint(self.cfg.run.save_dir / 'checkpoints/best.pth')
                print()

        metrics = ['epoch'] + list(results.keys())
        vals = [self.current_epoch] + list(results.values())
        self._save_results_csv(metrics, vals)

    @rank_zero_only
    def _save_results_csv(self, metrics, vals):
        csv_path = self.cfg.run.save_dir / 'results.csv'
        if self.rank == 0:
            s = '' if csv_path.exists() else (('%13s,' * (len(metrics)) % tuple(['epoch'] + metrics)).rstrip(',') + '\n')  # header
            with open(csv_path, 'a') as f:
                f.write(s + ('%13.5g,' * (len(metrics)) % tuple(vals)).rstrip(',') + '\n')

    @rank_zero_only
    def save_checkpoint(self, filepath):
        if self.rank == 0:
            if self.strategy == 'ddp':
                torch.save(self.model.module.state_dict(), filepath)
            else:
                torch.save(self.model.state_dict(), filepath)
