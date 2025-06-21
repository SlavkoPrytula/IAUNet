from configs import cfg
from . import get_model, load_weights, save_model_files

__all__ = ['build_model']


def build_model(cfg: cfg):
    model = get_model(cfg)
    
    if cfg.model.load_pretrained:
        model = load_weights(model, weights_path=cfg.model.weights)
    if cfg.model.save_model_files:
        save_model_files(model_cfg=cfg.model, save_dir=cfg.run.save_dir)
    
    return model
