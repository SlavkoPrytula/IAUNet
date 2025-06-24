from os.path import join, isfile
import importlib.util
import sys
import uuid
import modulefinder
import torch

from types import ModuleType
from typing import Type

import sys
sys.path.append("./")

from utils.files import _copy_folder
from configs import cfg, MODEL_FILES, CONFIG_FILES, UTILS_FILES, PROJECT_ROOT
from utils.registry import MODELS, DECODERS, HEADS


__all__ = ["get_model", "load_weights", "save_model_files"]


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


def load_weights(model, ckpt_path):
    print(f"- Loading pretrained weights:\n[{ckpt_path}]")

    current_model_dict = model.state_dict()
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    for k, v in state_dict.items():
        if k in current_model_dict and v.size() == current_model_dict[k].size():
            current_model_dict[k] = v
        elif k in current_model_dict:
            print(f"WARNING: Skipping loading weights for parameter '{k}' due to size mismatch.")
            print(f"Expected size: {current_model_dict[k].size()}, but got size: {v.size()}")
        else:
            if 'total_params' not in k and 'total_ops' not in k:
                print(f"WARNING: Skipping loading weights for parameter '{k}' as it was not found in the current model.")
    
    # Warn about weights in the current model but not present in the pretrained weights.
    for k in current_model_dict.keys():
        if k not in state_dict:
            print(f"WARNING: Parameter '{k}' in the current model is not present in the pretrained weights.")

    model.load_state_dict(current_model_dict, strict=False)

    print("- Weights loaded!")
    print()
    
    return model


def get_model_from_path(cfg: cfg):
    # runs/.../iaunet.py
    model_file = f"{cfg.model.model_files}/__init__.py"
    assert isfile(model_file), FileNotFoundError(f"Model file not found: {model_file}")

    print("Loading model from path...")
    print(f"Found model files: "
            f"\n- {cfg.model.model_files}")
    print()

    # import the module to register the model from model_files
    module = import_from_file(model_file, clear_cache=True)
    model = MODELS.get(cfg.model.type)(cfg=cfg)

    return model



def save_model_files(model_cfg, save_dir):
    """
    Save all relevant model, augmentation, and config files for reproducibility.
    """
    # Define what to copy: (source, destination_subdir, base_dir)
    utils_save_dir = save_dir / "utils"
    model_save_dir = save_dir / "model_files"

    copy_tasks = [
        # Augmentations
        (join(UTILS_FILES, 'augmentations', "augmentations.py"), utils_save_dir, UTILS_FILES),
        # Model files
        (join(MODEL_FILES, "__init__.py"), model_save_dir, MODEL_FILES),
        (join(MODEL_FILES, "losses"), model_save_dir, MODEL_FILES),
        (join(MODEL_FILES, "heads"), model_save_dir, MODEL_FILES),
        (join(MODEL_FILES, "nn/blocks"), model_save_dir, MODEL_FILES),
        (join(MODEL_FILES, "encoders/__init__.py"), model_save_dir, MODEL_FILES),
        (join(MODEL_FILES, "decoders/__init__.py"), model_save_dir, MODEL_FILES),
        # Dynamic model/encoder/decoder files
        (MODELS.get_path(model_cfg.type), model_save_dir, MODEL_FILES),
        (MODELS.get_path(model_cfg.encoder.type), model_save_dir, MODEL_FILES),
        (DECODERS.get_path(model_cfg.decoder.type), model_save_dir, MODEL_FILES),
    ]

    for src, dst, base_src_dir in copy_tasks:
        try:
            _copy_folder(src=src, dst=dst, base_src_dir=base_src_dir)
        except Exception as e:
            print(f"Warning: Could not copy {src} to {dst}: {e}")

    

if __name__ == "__main__":
    model_file = MODELS.get_path(cfg.model.type)
    print(model_file)
    dependent_files = find_module_dependencies(model_file, MODEL_FILES)
    for i, f in enumerate(dependent_files):
        print(f)
