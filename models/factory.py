from .utils import get_model, load_weights, save_model_files
from configs import cfg

__all__ = ['build_model']


def build_model(cfg: cfg):
    """
    Build the model based on the configuration. 
    This function initializes the model, loads pretrained weights if specified, and saves model files if required.
    When `cfg.model.load_from_files` is set to True, it will import the model from the specified path in `cfg.model.model_files`.

    Args:
        cfg (cfg): Configuration object containing model parameters and settings.
    Returns:
        model: The initialized model instance.
    """

    model = get_model(cfg)
    
    if cfg.model.load_pretrained:
        model = load_weights(model, ckpt_path=cfg.model.ckpt_path)
    if cfg.model.save_model_files:
        save_model_files(model_cfg=cfg.model, save_dir=cfg.run.save_dir)
    
    return model

