import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.rank_zero import rank_zero_only
from tools.get_flops import get_flops
from configs import cfg

from .base import BaseTrainer
from .loops import TrainLoop, ValidLoop, EvalLoop


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
            eval_dataloader,
            evaluators, 
            callbacks, 
            logger, 
            rank, 
            strategy,
            sync_batchnorm
    ) -> None:
        super().__init__(
            cfg, 
            model, 
            criterion, 
            optimizer, 
            scheduler, 
            train_dataloader, 
            valid_dataloader, 
            eval_dataloader,
            evaluators, 
            callbacks, 
            logger, 
            rank, 
            strategy,
            sync_batchnorm
        )

        self.model.to(self.device)
        if self.strategy == 'ddp':
            if self.sync_batchnorm:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[self.rank])

        # TODO: ProfileModelCallback - on_init
        get_flops(model, device=self.device)

        self.train_loop = TrainLoop(
            cfg=self.cfg,
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            dataloader=self.train_dataloader,
            device=self.device,
            logger=self.logger,
            evaluators=self.evaluators,
            callbacks=self.callbacks,
        )
        self.train_loop.trainer = self

        self.valid_loop = ValidLoop(
            cfg=self.cfg,
            model=self.model,
            criterion=self.criterion,
            dataloader=self.valid_dataloader,
            device=self.device,
            logger=self.logger,
            evaluators=self.evaluators,
            callbacks=self.callbacks,
        )
        self.valid_loop.trainer = self

        self.eval_loop = EvalLoop(
            cfg=self.cfg,
            model=self.model,
            criterion=self.criterion,
            dataloader=self.eval_dataloader,
            device=self.device,
            logger=self.logger,
            evaluators=self.evaluators,
            callbacks=self.callbacks,
        )
        self.eval_loop.trainer = self


    def train(self):
        start = time.time()

        for epoch in range(self.max_epochs):
            self.logger.info(f'Epoch {epoch}/{self.max_epochs}')
            self.current_epoch = epoch
            results_train = self.train_loop.run()
            
            if epoch % self.check_val_every_n_epoch == 0:
                results_valid = self.valid_loop.run()
                # results_eval = self.eval_loop.run()
                self._update_results(results_train, results_valid)

        end = time.time()
        time_elapsed = end - start
        self.logger.info('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
        

    def test(self):
        model = self.get_model('best')
        model.to(self.device)

        if self.strategy == 'ddp':
            model = DDP(model, device_ids=[self.rank])

        self.logger.info("Evaluating the best model...")
        self.eval_loop.model = model
        self.eval_loop.run()


    def _update_results(self, train_results, valid_results):
        results = {}
        results.update(valid_results)
        results.update(train_results)

        val_loss = results["loss_valid"]
        if val_loss <= self.best_loss:
            self.logger.info(f"Valid Loss Improved ({self.best_loss:.4f} ---> {val_loss:.4f})")
            self.best_loss = val_loss
            
            # saving best model.
            self.save_checkpoint(self.cfg.run.save_dir / 'checkpoints/best.pth')
            print()

        # saving last checkpoint.
        self.save_checkpoint(self.cfg.run.save_dir / 'checkpoints/last.pth')

        metrics = ['epoch'] + list(results.keys())
        vals = [self.current_epoch] + list(results.values())
        self._save_results_csv(metrics, vals)

    @rank_zero_only
    def _save_results_csv(self, metrics, vals):
        csv_path = self.cfg.run.save_dir / 'results.csv'
        s = '' if csv_path.exists() else (('%13s,' * (len(metrics)) % tuple(metrics)).rstrip(',') + '\n')  # header
        with open(csv_path, 'a') as f:
            f.write(s + ('%13.5g,' * (len(metrics)) % tuple(vals)).rstrip(',') + '\n')

    @rank_zero_only
    def save_checkpoint(self, filepath):
        if not self.cfg.model.save_checkpoint:
            return 
        
        if self.strategy == 'ddp':
            torch.save(self.model.module.state_dict(), filepath)
        else:
            torch.save(self.model.state_dict(), filepath)

    def get_model(self, model_type='best'):
        from models import load_weights

        if model_type == 'best':
            checkpoint_path = self.cfg.run.save_dir / 'checkpoints/best.pth'
        elif model_type == 'latest':
            checkpoint_path = self.cfg.run.save_dir / 'checkpoints/last.pth'
        else:
            raise ValueError(f"Unknown model type '{model_type}'")

        print(f'Loading "{model_type}" weights from {checkpoint_path}...')
        model = self.model.__class__(self.cfg)
        model = load_weights(model, checkpoint_path)
        
        return model