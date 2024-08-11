from .registry import Registry
from typing import Any
from utils import visualise

from .registry import build_from_cfg
from omegaconf import OmegaConf, DictConfig
from configs import cfg
from configs.structure import Callbacks, Visualizer


# def build_optimizer(cfg: cfg) -> Optimizer:
#     name = cfg.type
#     cfg.pop("type")
#     return OPTIMIZERS.get(name)()(**cfg)


# def build_scheduler(cfg: cfg) -> lr_scheduler:
#     name = cfg.type
#     cfg.pop("type")
#     return SCHEDULERS.get(name)(**cfg)


# def build_from_cfg(cfg: cfg, registry: Registry) -> Any:
#     name = cfg.get('type')
#     cfg.pop("type")
#     return registry.get(name)()(**cfg)


# def build_matcher(cfg: cfg, registry: Registry):
#     name = cfg.model.criterion.matcher.type
#     return registry.get(name)(cfg)


# def build_criterion(cfg: cfg, registry: Registry):
#     from . import MATCHERS
#     matcher = build_matcher(cfg, MATCHERS)
#     name = cfg.model.criterion.type
#     return registry.get(name)(cfg, matcher)


def build_matcher(cfg: cfg, registry: Registry=None) -> Any:
    # scope switch
    if registry is None:
        from . import MATCHERS
        registry = MATCHERS

    name = cfg.get('type')
    return registry.get(name)(cfg)


def build_criterion(cfg: cfg, registry: Registry=None) -> Any:
    # scope switch
    if registry is None:
        from . import CRITERIONS
        registry = CRITERIONS

    matcher = build_matcher(cfg.matcher)
    name = cfg.get('type')
    return registry.get(name)(cfg, matcher)


def build_optimizer(cfg: cfg, registry: Registry=None) -> Any:
    # scope switch
    if registry is None:
        from . import OPTIMIZERS
        registry = OPTIMIZERS

    # return build_from_cfg(cfg, registry)
    name = cfg.get('type')
    cfg.pop("type")
    return registry.get(name)()(**cfg)


def build_scheduler(cfg: cfg, registry: Registry=None) -> Any:
    # scope switch
    if registry is None:
        from . import SCHEDULERS
        registry = SCHEDULERS
        
    # return build_from_cfg(cfg, registry)
    name = cfg.get('type')
    cfg.pop("type")
    return registry.get(name)()(**cfg)



def build_visualizer(cfg: Visualizer, registry: Registry=None) -> Any:
    name = cfg.get('type')

    if isinstance(cfg, dict):
        _cfg = cfg.copy()
        _cfg.pop("type")
        return registry.get(name)(**_cfg)
    
    _cfg = OmegaConf.to_container(cfg, resolve=True)
    _cfg.pop("type", None)

    return registry.get(name)(**_cfg)


def build_callback(cfg, registry: Registry=None) -> Any:
    # scope switch
    if registry is None:
        from . import CALLBACKS
        registry = CALLBACKS

    name = cfg.get('type')
    # we are hacking...
    if 'Visualizer' in name:
        return build_visualizer(cfg, registry)
    
    return build_from_cfg(cfg, registry)


def build_decoder(cfg: cfg, registry: Registry=None, **kwargs) -> Any:
    if registry is None:
        from . import DECODERS
        registry = DECODERS
        
    name = cfg.get('type')
    if isinstance(cfg, dict):
        _cfg = cfg.copy()
    elif isinstance(cfg, OmegaConf) or isinstance(cfg, DictConfig):
        _cfg = OmegaConf.to_container(cfg, resolve=True)
    
    _cfg.pop("type", None)
    return registry.get(name)(cfg=cfg, **kwargs)