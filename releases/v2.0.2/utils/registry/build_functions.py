from configs import cfg
from .registry import Registry
from typing import Any


# def build_optimizer(cfg: cfg) -> Optimizer:
#     name = cfg.type
#     cfg.pop("type")
#     return OPTIMIZERS.get(name)()(**cfg)


# def build_scheduler(cfg: cfg) -> lr_scheduler:
#     name = cfg.type
#     cfg.pop("type")
#     return SCHEDULERS.get(name)(**cfg)


def build_from_cfg(cfg: cfg, registry: Registry) -> Any:
    name = cfg.type
    cfg.pop("type")
    return registry.get(name)()(**cfg)


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

    name = cfg.type
    return registry.get(name)(cfg)


def build_criterion(cfg: cfg, registry: Registry=None) -> Any:
    # scope switch
    if registry is None:
        from . import CRITERIONS
        registry = CRITERIONS

    matcher = build_matcher(cfg.matcher)
    name = cfg.type
    return registry.get(name)(cfg, matcher)


def build_optimizer(cfg: cfg, registry: Registry=None) -> Any:
    # scope switch
    if registry is None:
        from . import OPTIMIZERS
        registry = OPTIMIZERS

    return build_from_cfg(cfg, registry)


def build_scheduler(cfg: cfg, registry: Registry=None) -> Any:
    # scope switch
    if registry is None:
        from . import SCHEDULERS
        registry = SCHEDULERS
        
    return build_from_cfg(cfg, registry)


