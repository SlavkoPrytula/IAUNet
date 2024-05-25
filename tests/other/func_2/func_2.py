# from ..func_1 import func_1
import os
import importlib
import sys
from types import ModuleType
from typing import Type, cast
from typing_extensions import Protocol


def func_1(x: int): int


def import_from_file(module_name, file_path) -> Type[ModuleType]:
    """
    reads file source and loads it as a module

    :param source: file to load
    :param module_name: name of module to register in sys.modules
    :return: loaded module
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    print(module_name in sys.modules)
    sys.modules[module_name] = module
    print(module_name in sys.modules)
    spec.loader.exec_module(module)

    return module



mapping = {
    'func_1': 'test/func_1/func_1.py'
}


module = import_from_file('func_1', mapping['func_1'])

func_1 = cast(func_1, getattr(module, 'func_1'))

print(func_1(2))

current_directory = os.path.dirname(os.path.realpath(__file__))
print(current_directory)