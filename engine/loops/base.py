from configs import cfg
from engine.trainer import BaseTrainer


class BaseLoop:
    def __init__(
            self, 
            cfg: cfg, 
            model, 
            criterion, 
            dataloader, 
            device, 
            logger, 
            callbacks, 
            evaluators, 
            **kwargs
    ):
        self.trainer: BaseTrainer = None
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.dataloader = dataloader
        self.device = device
        self.logger = logger
        self.callbacks = callbacks
        self.evaluators = evaluators

        self.total_steps = len(self.dataloader) if self.dataloader else None

        for key, value in kwargs.items():
            setattr(self, key, value)

    def trigger_callbacks(self, event, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, event):
                getattr(callback, event)(**kwargs)

    @property
    def epoch(self):
        return self.trainer.current_epoch