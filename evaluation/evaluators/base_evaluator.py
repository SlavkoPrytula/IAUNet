import torch
from torch import nn
from configs import cfg
from abc import ABC, abstractmethod


class BaseEvaluator(nn.Module, ABC):
    def __init__(self, cfg: cfg, model=None, **kwargs):
        super(BaseEvaluator, self).__init__()
        self.cfg = cfg
        self.model = model
        if model:
            self.model.eval()
        else:
            print("WARNING: model is None, model should be correctly passed to the Evaluator")
        self.device = next(model.parameters()).device
        self.stats = None

    @abstractmethod
    def forward(self, *args, **kwargs):
        if not callable(getattr(self.model, "inference", None)):
            print("UserWarning: In the new release v2.1.0 model classes should have inference methods!")
        ...

    @abstractmethod
    def process(self, preds: dict):
        ...

    @torch.no_grad()
    def inference_single(self, input):
        output = self.model(input)
        return output

    @abstractmethod
    def evaluate(self, verbose=False):
        ...