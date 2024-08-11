from .registry import Registry
from .build_functions import (build_matcher, build_criterion, build_callback, 
                              build_optimizer, build_scheduler, build_decoder)


MODELS = Registry("model")
HEADS = Registry("head")
DECODERS = Registry("decoder", build_func=build_decoder)
CRITERIONS = Registry("criterion", build_func=build_criterion)
MATCHERS = Registry("matcher", build_func=build_matcher)

OPTIMIZERS = Registry("optimizer", build_func=build_optimizer)
SCHEDULERS = Registry("scheduler", build_func=build_scheduler)

DATASETS = Registry("dataset")
DATASETS_CFG = Registry("datasets_cfg")

EVALUATORS = Registry("evaluator")
METRICS = Registry("metric")

CALLBACKS = Registry("callbacks", build_func=build_callback)
