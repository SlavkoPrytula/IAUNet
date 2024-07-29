from os.path import join, isfile
import importlib.util
import sys
import uuid
import modulefinder

from types import ModuleType
from typing import Type, cast
from typing_extensions import Protocol

import sys
sys.path.append("./")
import torch
from models.seg import *
from utils.files import _copy_folder
from configs import cfg, MODEL_FILES, CONFIG_FILES, UTILS_FILES
from utils.registry import MODELS


def import_from_file(file_path, clear_cache=False) -> Type[ModuleType]:
    """
    Dynamically load module from file

    :param file_path: file to load
    :return: loaded module
    """
    # Work around on module reloading, importing the new module 
    # under a unique name and removing the old one from sys cache
    module_name = str(uuid.uuid4())  # Generate a unique module name
    if clear_cache and module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    return module


def find_module_dependencies(module_file, base_dir):
    import sys
    from configs import PROJECT_DIR
    sys.path.append(PROJECT_DIR)

    finder = modulefinder.ModuleFinder()
    finder.run_script(module_file)

    dependent_files = set()
    for name, mod in finder.modules.items():
        if mod.__file__ and mod.__file__.startswith(base_dir):
            dependent_files.add(mod.__file__)
    
    return dependent_files


def get_model(cfg: cfg):
    if cfg.model.load_from_files:
        model = get_model_from_path(cfg)
    else:
        model = MODELS.get(cfg.model.type)(cfg=cfg)
    
    return model


def load_weights(model, weights_path):
    print(f"- Loading pretrained weights:\n[{weights_path}]")

    current_model_dict = model.state_dict()
    loaded_state_dict = torch.load(weights_path) #["state_dict"]

    for k, v in loaded_state_dict.items():
        if k in current_model_dict and v.size() == current_model_dict[k].size():
            current_model_dict[k] = v
        elif k in current_model_dict:
            print(f"WARNING: Skipping loading weights for parameter '{k}' due to size mismatch.")
            print(f"Expected size: {current_model_dict[k].size()}, but got size: {v.size()}")
        else:
            print(f"WARNING: Skipping loading weights for parameter '{k}' as it was not found in the current model.")
    
    # Warn about weights in the current model but not present in the pretrained weights.
    for k in current_model_dict.keys():
        if k not in loaded_state_dict:
            print(f"WARNING: Parameter '{k}' in the current model is not present in the pretrained weights.")

    model.load_state_dict(current_model_dict, strict=False)

    print("- Weights loaded!")
    
    return model


def get_model_from_path(cfg: cfg):
    # runs/.../iaunet.py
    model_file = f"{cfg.model.model_files}/__init__.py"
    assert isfile(model_file), FileNotFoundError(f"Model file not found: {model_file}")

    if cfg.verbose: 
        print("Loading model from path...")
        print(f"Found model files: "
              f"\n- {cfg.model.model_files}")

    # import the module to register the model from model_files
    module = import_from_file(model_file, clear_cache=True)
    model = MODELS.get(cfg.model.type)(cfg=cfg)
    
    return model


def save_model_files(model_cfg, save_dir):    
    # save config files
    config_dst = save_dir / "config_files"
    _copy_folder(
        src=join(CONFIG_FILES, "base.py"),
        dst=config_dst,
        base_src_dir=CONFIG_FILES
    )

    # save augmentation files
    aug_dst = save_dir / "utils"
    _copy_folder(
        src=join(UTILS_FILES, 'augmentations', "augmentations.py"),
        dst=aug_dst,
        base_src_dir=UTILS_FILES
    )

    # save model file (eg. models, heads, losses, ...)
    model_dst = save_dir / "model_files"
    for src in ["__init__.py", "matcher.py", "loss.py", 
                "heads", "models/__init__.py", "nn/blocks", 
                "encoders/__init__.py"]:
        _copy_folder(
            src=join(MODEL_FILES, src),
            dst=model_dst,
            base_src_dir=MODEL_FILES
            )
    
    # save model files.
    model_file = MODELS.get_path(model_cfg.type)
    _copy_folder(
        src=model_file,
        dst=model_dst,
        base_src_dir=MODEL_FILES
        )
    
    # save encoder files.
    model_file = MODELS.get_path(model_cfg.backbone.type)
    _copy_folder(
        src=model_file,
        dst=model_dst,
        base_src_dir=MODEL_FILES
        )
    


if __name__ == "__main__":
    model_file = MODELS.get_path(cfg.model.type)
    print(model_file)
    dependent_files = find_module_dependencies(model_file, MODEL_FILES)
    for i, f in enumerate(dependent_files):
        print(f)