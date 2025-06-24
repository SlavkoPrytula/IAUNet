from torch import nn
from configs import cfg
from abc import ABC, abstractmethod


class BaseEvaluator(nn.Module, ABC):
    def __init__(self, cfg: cfg, **kwargs):
        super(BaseEvaluator, self).__init__()
        self.cfg = cfg
        self.stats = None

    @abstractmethod
    def process(self, preds: dict):
        ...

    @abstractmethod
    def evaluate(self, verbose=False):
        ...