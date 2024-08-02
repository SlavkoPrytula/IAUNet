import time
import torch
import numpy as np
from pathlib import Path
from configs import cfg

from .train import train_one_epoch
from .valid import valid_one_epoch


class Trainer:
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
            rank=0
    ):
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.evaluators = evaluators
        self.callbacks = callbacks
        self.logger = logger
        self.device = self._get_device()
        self.best_loss = np.inf
        self.rank = rank

        # makedirs(cfg.run.save_dir / 'checkpoints', exist_ok=True)
        self.csv_path = self.cfg.run.save_dir / 'results.csv'

        self.train_loop = train_one_epoch
        self.valid_loop = valid_one_epoch


    def _get_device(self):
        rank = self.rank
        if self.cfg.trainer.accelerator == 'gpu' and torch.cuda.is_available():
            device = torch.device(f'cuda:{rank}' if rank is not None else 'cuda')
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(device)} on rank {rank}")
        else:
            device = torch.device('cpu')
            self.logger.info("Using CPU\n")
        return device


    def train(self):
        start = time.time()
        num_epochs = self.cfg.trainer.max_epochs + 1

        for epoch in range(num_epochs):
            self.logger.info(f'Epoch {epoch}/{num_epochs}')

            results_train = self.train_loop(cfg, 
                                            model=self.model, 
                                            criterion=self.criterion, 
                                            optimizer=self.optimizer, 
                                            scheduler=self.scheduler,
                                            dataloader=self.train_dataloader,
                                            device=self.device, 
                                            epoch=epoch,
                                            callbacks=self.callbacks,
                                            logger=self.logger,
                                            )
            
            if epoch % cfg.trainer.check_val_every_n_epoch == 0:
                results_valid = self.valid_loop(cfg, 
                                                model=self.model, 
                                                criterion=self.criterion, 
                                                optimizer=self.optimizer, 
                                                scheduler=self.scheduler,
                                                dataloader=self.valid_dataloader,
                                                device=self.device, 
                                                epoch=epoch,
                                                evaluators=self.evaluators,
                                                callbacks=self.callbacks,
                                                logger=self.logger,
                                                )


                self._update_results(results_train, results_valid, epoch)


        # saving last checkpoint.
        if cfg.model.save_checkpoint:
            self.save_checkpoint(self.cfg.run.save_dir / 'checkpoints/last.pth')

        end = time.time()
        time_elapsed = end - start
        self.logger.info('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))


    def _update_results(self, train_results, valid_results, epoch):
        results = {}
        results.update(valid_results)
        results.update(train_results)

        val_loss = results["loss_valid"]
        if val_loss <= self.best_loss:
            self.logger.info(f"Valid Loss Improved ({self.best_loss:.4f} ---> {val_loss:.4f})")
            self.best_loss = val_loss
            
            # saving best model.
            if cfg.model.save_checkpoint:
                self.save_checkpoint(self.cfg.run.save_dir / 'checkpoints/best.pth')
                print()

        metrics = ['epoch'] + list(results.keys())
        vals = [epoch] + list(results.values())
        self._save_results_csv(metrics, vals)


    def _save_results_csv(self, metrics, vals):
        s = '' if self.csv_path.exists() else (('%13s,' * (len(metrics)) % tuple(['epoch'] + metrics)).rstrip(',') + '\n')  # header
        with open(self.csv_path, 'a') as f:
            f.write(s + ('%13.5g,' * (len(metrics)) % tuple(vals)).rstrip(',') + '\n')


    def save_checkpoint(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    @property
    def rank(self) -> int:
        return self.rank if self.rank else 0