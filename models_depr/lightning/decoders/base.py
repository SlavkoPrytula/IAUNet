import torch
from torch import nn
from abc import ABC

import sys
sys.path.append("./")

from configs.structure import Decoder



class BaseDecoder(nn.Module, ABC):
    def __init__(self, cfg: Decoder, embed_dims: list = [], n_levels: int = 4):
        super(BaseDecoder, self).__init__()
        
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

    def forward(self, skips, ori_shape):
        """
        Default decoder forward function:
        - _forward() -> dict()
        - process_outputs() -> dict()

        ori_shape - max_shape of an image to be returned
        """
        results = self._forward(skips)
        results = self.process_outputs(results, ori_shape)
        return results

    def _forward(self, features):
        ...

    def process_outputs(self, results, ori_shape):
        ...
