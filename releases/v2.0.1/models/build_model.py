from os.path import join, dirname, isfile
from os import makedirs
import shutil

import torch
from torch.nn.parallel import DistributedDataParallel

from utils import comm
from configs import cfg

from . import get_model, load_weights, save_model_files

__all__ = [
    'build_model'
    ]


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    if comm.get_world_size() == 1:
        return model
    
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp



def build_model(cfg: cfg):
    model = get_model(cfg)
    
    # TODO: do this inside the model class (every model might have different weights mapping)
    # NOTE: moved weights initialization to model class
    if cfg.model.load_pretrained:
        model = load_weights(model, weights_path=cfg.model.weights)
        # try:
        #     model = load_weights(model, weights_path=cfg.model.weights)
        # except:
        #     model.init_weights()

    # DEBUG: save model files
    if cfg.model.save_model_files:
        save_model_files(arch=cfg.model.arch, save_dir=cfg.save_dir)

    model.to(cfg.device)

    return model


# NOTE: Deprecated function 
def load_model(cfg: cfg, path: str = None):
    model = build_model(cfg)
    # model.load_state_dict(torch.load(path), strict=False)
    # model.to(cfg.device)
    model.eval()
    print('- weights loaded!')

    return model


if __name__ == '__main__':
    from configs.base import cfg
    model = build_model(cfg)

    x = torch.randn(1, 1, 128, 128).to(cfg.device)
    out = model(x)
    print(out.shape)
