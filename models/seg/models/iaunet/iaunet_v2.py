# copied from 45966022
import torch
from torch.nn import functional as F

import sys
sys.path.append("./")

from models.seg.models.base import BaseModel
from configs import cfg
from utils.registry import MODELS, DECODERS



@MODELS.register(name="iaunet_v2")
class IAUNet(BaseModel):
    def __init__(self, cfg: cfg):
        super(IAUNet, self).__init__(cfg)

        self.encoder = MODELS.build(cfg.model.encoder)
        embed_dims = self.encoder.embed_dims
        self.embed_dims = embed_dims

        self.decoder = DECODERS.build(cfg.model.decoder, 
                                      embed_dims=self.embed_dims) 

    def forward(self, x):
        max_shape = x.shape[-2:]

        skips = self.encoder(x)
        results = self.decoder(skips, max_shape)
        
        return results
