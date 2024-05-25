from .registry import Registry
from .build_functions import build_from_cfg, build_matcher, build_criterion


MODELS = Registry("model")
CRITERIONS = Registry("criterion", build_func=build_criterion)
MATCHERS = Registry("matcher", build_func=build_matcher)

OPTIMIZERS = Registry("optimizer")
SCHEDULERS = Registry("scheduler")

DATASETS = Registry("dataset")

EVALUATORS = Registry("evaluator")
