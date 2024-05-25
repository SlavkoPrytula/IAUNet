import torch
from torch import nn
from configs import cfg

from abc import ABC, abstractmethod

    
class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x   
    
    
class SE_block(nn.Module):
    """squeeze and excitation block"""
    def __init__(self, num_features, reduction_factor=2):
        super(SE_block, self).__init__()
        # squeeze block
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # excitation block
        self.excite = nn.Sequential(
            nn.Linear(num_features, num_features // reduction_factor),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // reduction_factor, num_features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch, channel, _, _ = x.size()
        squeeze_res = self.squeeze(x).view(batch, channel)
        excite_res = self.excite(squeeze_res)
        f_scale = excite_res.view(batch, channel, 1, 1)
        return x * f_scale



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
