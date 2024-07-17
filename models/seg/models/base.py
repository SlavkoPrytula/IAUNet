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
        self.n_levels = cfg.model.n_levels

        self.coord_conv = cfg.model.coord_conv
        self.multi_level = cfg.model.multi_level
        self.kernel_dim = cfg.model.instance_head.kernel_dim
        self.num_convs = cfg.model.num_convs
        self.mask_dim = cfg.model.mask_dim

        
        
    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        x_loc = torch.linspace(-1, 1, h, device=x.device)
        y_loc = torch.linspace(-1, 1, w, device=x.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x_loc, y_loc], 1)

        return coord_feat
        

    @abstractmethod
    def forward(self, x):
        ...


    # @abstractmethod
    # def inference(self, x):
    #     ...
