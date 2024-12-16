import torch
from torch import nn
from configs import cfg

from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    def __init__(
        self,
        cfg: cfg,
    ):
        super(BaseModel, self).__init__()
        self.cfg = cfg  
        
    @abstractmethod
    def forward(self, x):
        ...

    # @abstractmethod
    # def inference(self, x):
    #     ...
