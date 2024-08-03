from os.path import join, dirname, isfile
from os import makedirs
import shutil

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.comm import get_world_size, get_local_rank
from configs import cfg

from . import get_model, load_weights, save_model_files

__all__ = [
    'build_model'
    ]


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    if get_world_size() == 1:
        return model
    
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [get_local_rank()]
    ddp = DDP(model, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp



# def build_model(cfg: cfg, rank=None, world_size=None):
#     model = get_model(cfg)
#     print(model)
    
#     # TODO: do this inside the model class (every model might have different weights mapping)
#     # NOTE: moved weights initialization to model class
#     if cfg.model.load_pretrained:
#         model = load_weights(model, weights_path=cfg.model.weights)

#     if cfg.model.save_model_files:
#         save_model_files(model_cfg=cfg.model, save_dir=cfg.save_dir)

#     # device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
#     model.to(cfg.device)
    
#     # if rank is not None:
#     #     model = create_ddp_model(model, device_ids=[rank])

#     return model



def build_model(cfg: cfg):
    model = get_model(cfg)
    
    if cfg.model.load_pretrained:
        model = load_weights(model, weights_path=cfg.model.weights)
    if cfg.model.save_model_files:
        save_model_files(model_cfg=cfg.model, save_dir=cfg.run.save_dir)

    # if cfg.trainer.accelerator == 'gpu' and torch.cuda.is_available():
    #     device = torch.device(f'cuda:{rank}' if rank is not None else 'cuda')
    # else:
    #     device = torch.device('cpu')

    # print(f'build_model: {device}')

    # if cfg.trainer.get('strategy') == 'ddp' and rank is not None:
    #     print('running ddp')
    #     model = create_ddp_model(model)
    # elif cfg.trainer.get('strategy') == 'dp' and torch.cuda.device_count() > 1:
    #     print('running dp')
    #     model = torch.nn.DataParallel(model)
    
    # device = "cuda"
    # model.to(device)
    
    return model

