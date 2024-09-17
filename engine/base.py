import torch
import numpy as np
import os
from configs import cfg


class BaseTrainer:
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
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.eval_dataloader = eval_dataloader
        self.evaluators = evaluators
        self.callbacks = callbacks
        self.logger = logger
        self.rank = rank
        self.strategy = strategy
        self.sync_batchnorm = sync_batchnorm
        self.device = self._get_device()

        self.train_loop = None
        self.valid_loop = None
        self.eval_loop = None

        self.max_epochs = cfg.trainer.max_epochs + 1
        self.best_metric = -np.inf
        self.loss = None
        self.loss_dict = None
        self.output = None
        self.current_epoch = None

        self.check_val_every_n_epoch = cfg.trainer.check_val_every_n_epoch

        print(f"Initializing Trainer at RANK: {int(os.getenv('RANK', 0))}.")
        print(f"Running with {self.strategy} strategy.")
        print()

        self._setup_evaluators()

    def _get_device(self):
        rank = self.rank
        if self.cfg.trainer.accelerator == 'gpu' and torch.cuda.is_available():
            device = torch.device(f'cuda:{rank}' if rank is not None else 'cuda')
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(device)} on rank {rank}")
        else:
            device = torch.device('cpu')
            self.logger.info("Using CPU\n")
        return device
    
    def _setup_evaluators(self):
        for stage in self.evaluators:
            for evaluator in self.evaluators[stage]:
                self.evaluators[stage][evaluator].device = self.device


    def train(self):
        ...

    def test(self):
        ...