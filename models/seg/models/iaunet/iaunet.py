# copied from 45966022
import torch
from torch.nn import functional as F

import sys
sys.path.append("./")

from models.seg.models.base import BaseModel
from configs import cfg
from utils.registry import MODELS, DECODERS



@MODELS.register(name="iaunet")
class IAUNet(BaseModel):
    def __init__(self, cfg: cfg):
        super(IAUNet, self).__init__(cfg)

        self.encoder = MODELS.build(cfg.model.encoder)
        embed_dims = self.encoder.embed_dims
        self.embed_dims = embed_dims

        self.decoder = DECODERS.build(cfg.model.decoder, 
                                      embed_dims=self.embed_dims,
                                      n_levels=self.n_levels)

    def forward(self, x):
        ori_shape = x.shape

        skips = self.encoder(x)
        results = self.decoder(skips, ori_shape)
        
        return results
