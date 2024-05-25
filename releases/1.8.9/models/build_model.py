from os.path import join, dirname, isfile
from os import makedirs
import shutil

import torch

from configs import cfg

from . import get_model, load_weights, save_model_files


def build_model(cfg: cfg):
    model = get_model(cfg)
    
    if cfg.model.load_pretrained:
       model = load_weights(model, weights_path=cfg.model.weights)

    # DEBUG: save model files
    if cfg.model.save_model_files:
        save_model_files(arch=cfg.model.arch, save_dir=cfg.save_dir)

    model.to(cfg.device)

    return model


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
