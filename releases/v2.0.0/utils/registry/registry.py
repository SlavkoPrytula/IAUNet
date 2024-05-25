import copy
import logging
import types
from collections import UserDict
from typing import Any, Dict, List, Iterable, Iterator, Tuple, Generator, Optional, Type, Union, Callable


def build_from_cfg(cfg, registry) -> Any:
    name = cfg.type
    cfg.pop("type")
    return registry.get(name)()(**cfg)

from tabulate import tabulate


__all__ = ["Registry"]


class Registry(Iterable[Tuple[str, Any]]):
    """
    A global dictionary that stores information about the datasets and how to obtain them.

    It contains a mapping from strings
    (which are names that identify a dataset, e.g. "coco_2014_train")
    to a function which parses the dataset and returns the samples in the
    format of `list[dict]`.

    The returned dicts should be in Detectron2 Dataset format (See DATASETS.md for details)
    if used with the data loader functionalities in `data/build.py,data/detection_transform.py`.

    The purpose of having this catalog is to make it easy to choose
    different datasets, by just using the strings in the config.
    """
    def __init__(self, 
                 name: str, 
                 build_func: Optional[Callable] = None
                 ):
        """
        Args:
            name (str): the name of this registry
        """
        # from .build_functions import build_from_cfg

        self._name: str = name
        self._module_dict: Dict[str, Any] = {}

        self.build_func = build_func
        if self.build_func is None:
            self.build_func = build_from_cfg

    def _register_module(self, module, module_name=None):
        if module_name is None:
                module_name = module.__name__
        self._module_dict[module_name] = module


    def register(self, name=None, module=None):
        """
        Args:
            name (str): the name that identifies a dataset, e.g. "coco_2014_train".
            func (callable): a callable which takes no arguments and returns a list of dicts.
                It must return the same results if called multiple times.
        """
        # assert callable(func), "You must register a function with `DatasetCatalog.register`!"
        # assert name not in self, "Module '{}' is already registered!".format(name)

        if module is not None:
            self._register_module(module=module, module_name=name)
            return module

        # used as a decorator
        def _register(module: Any) -> Any:
            self._register_module(module=module, module_name=name)
            return module

        return _register


    def get(self, name: str) -> Any:
        ret = self._module_dict.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(name, self._name)
            )
        return ret
    

    def build(self, cfg: dict) -> Any:
        """Build an instance.

        Build an instance by calling :attr:`build_func`.

        Args:
            cfg (dict): Config dict needs to be built.

        Returns:
            Any: The constructed object.
        """
        # assert isinstance(cfg, dict)
        return self.build_func(cfg, registry=self)


    def __contains__(self, name: str) -> bool:
        return name in self._module_dict


    def __repr__(self) -> str:
        table_headers = ["Names", "Objects"]
        table = tabulate(
            self._module_dict.items(), headers=table_headers, tablefmt="fancy_grid"
        )
        return "Registry of {}:\n".format(self._name) + table


    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        return iter(self._module_dict.items())

    # pyre-fixme[4]: Attribute must be annotated.
    __str__ = __repr__
