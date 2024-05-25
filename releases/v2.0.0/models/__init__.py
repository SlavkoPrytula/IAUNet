from os.path import join, dirname, isfile
from os import makedirs
import os.path 
import shutil
import importlib.util
import re
import sys
import uuid

from types import ModuleType
from typing import Type, cast
from typing_extensions import Protocol

import torch
from models.seg import *
from configs import cfg, MODEL_FILES

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


def get_model(cfg: cfg):
    # TODO: rewrite files loading for models -- done
    if cfg.model.load_from_files:
        model = get_model_from_path(cfg)
    else:
        model = MODELS.get(cfg.model.arch)(cfg=cfg)
    
    return model


def load_weights(model, weights_path):
    print(f"- Loading pretrained weights:\n[{weights_path}]")

    current_model_dict = model.state_dict()
    loaded_state_dict = torch.load(weights_path)

    
    module_mapping = {
            "down_conv_layers": [],
            "down_se_blocks": [],
            "down_pp_layers": [],
            "pp_se_blocks": [],

            "middleConv": [],
            "middleSE": [],

            # "up_se_blocks": ["up_se_blocks_occluders"],
            # "up_conv_layers": ["up_conv_layers_occluders"],

            # "mask_branch": ["mask_branch_occluder"], # , "out_mask_branch_occluder", "out_mask_branch"],
            # "prior_instance_branch": ["prior_occluder_branch"], #, "out_instance_branch", "out_occluder_branch"],
            # "instance_branch": ["occluder_branch"],
        }
    
    prior_instance_branch_mappings = {
        "prior_instance_branch.0.inst_convs.0":  ["prior_instance_branch.0.0.inst_convs.0", "prior_occluder_branch.0.0.inst_convs.0"],
        "prior_instance_branch.0.inst_convs.1":  ["prior_instance_branch.0.0.inst_convs.1", "prior_occluder_branch.0.0.inst_convs.1"],
        "prior_instance_branch.0.inst_convs.3":  ["prior_instance_branch.0.0.inst_convs.3", "prior_occluder_branch.0.0.inst_convs.3"],
        "prior_instance_branch.0.inst_convs.4":  ["prior_instance_branch.0.0.inst_convs.4", "prior_occluder_branch.0.0.inst_convs.4"],
        "prior_instance_branch.0.inst_convs.6":  ["prior_instance_branch.0.1.inst_convs.0", "prior_occluder_branch.0.1.inst_convs.0"],
        "prior_instance_branch.0.inst_convs.7":  ["prior_instance_branch.0.1.inst_convs.1", "prior_occluder_branch.0.1.inst_convs.1"],
        "prior_instance_branch.0.inst_convs.9":  ["prior_instance_branch.0.1.inst_convs.3", "prior_occluder_branch.0.1.inst_convs.3"],
        "prior_instance_branch.0.inst_convs.10": ["prior_instance_branch.0.1.inst_convs.4", "prior_occluder_branch.0.1.inst_convs.4"],
        
        "prior_instance_branch.0.inst_convs.12":  ["prior_instance_branch.0.2.inst_convs.0", "prior_occluder_branch.0.2.inst_convs.0"],
        "prior_instance_branch.0.inst_convs.13":  ["prior_instance_branch.0.2.inst_convs.1", "prior_occluder_branch.0.2.inst_convs.1"],
        "prior_instance_branch.0.inst_convs.15":  ["prior_instance_branch.0.2.inst_convs.3", "prior_occluder_branch.0.2.inst_convs.3"],
        "prior_instance_branch.0.inst_convs.16":  ["prior_instance_branch.0.2.inst_convs.4", "prior_occluder_branch.0.2.inst_convs.4"],
 

        "prior_instance_branch.1.inst_convs.0":  ["prior_instance_branch.1.0.inst_convs.0", "prior_occluder_branch.1.0.inst_convs.0"],
        "prior_instance_branch.1.inst_convs.1":  ["prior_instance_branch.1.0.inst_convs.1", "prior_occluder_branch.1.0.inst_convs.1"],
        "prior_instance_branch.1.inst_convs.3":  ["prior_instance_branch.1.0.inst_convs.3", "prior_occluder_branch.1.0.inst_convs.3"],
        "prior_instance_branch.1.inst_convs.4":  ["prior_instance_branch.1.0.inst_convs.4", "prior_occluder_branch.1.0.inst_convs.4"],
        "prior_instance_branch.1.inst_convs.6":  ["prior_instance_branch.1.1.inst_convs.0", "prior_occluder_branch.1.1.inst_convs.0"],
        "prior_instance_branch.1.inst_convs.7":  ["prior_instance_branch.1.1.inst_convs.1", "prior_occluder_branch.1.1.inst_convs.1"],
        "prior_instance_branch.1.inst_convs.9":  ["prior_instance_branch.1.1.inst_convs.3", "prior_occluder_branch.1.1.inst_convs.3"],
        "prior_instance_branch.1.inst_convs.10": ["prior_instance_branch.1.1.inst_convs.4", "prior_occluder_branch.1.1.inst_convs.4"],
        
        "prior_instance_branch.1.inst_convs.12":  ["prior_instance_branch.1.2.inst_convs.0", "prior_occluder_branch.1.2.inst_convs.0"],
        "prior_instance_branch.1.inst_convs.13":  ["prior_instance_branch.1.2.inst_convs.1", "prior_occluder_branch.1.2.inst_convs.1"],
        "prior_instance_branch.1.inst_convs.15":  ["prior_instance_branch.1.2.inst_convs.3", "prior_occluder_branch.1.2.inst_convs.3"],
        "prior_instance_branch.1.inst_convs.16":  ["prior_instance_branch.1.2.inst_convs.4", "prior_occluder_branch.1.2.inst_convs.4"],


        "prior_instance_branch.2.inst_convs.0":  ["prior_instance_branch.2.0.inst_convs.0", "prior_occluder_branch.2.0.inst_convs.0"],
        "prior_instance_branch.2.inst_convs.1":  ["prior_instance_branch.2.0.inst_convs.1", "prior_occluder_branch.2.0.inst_convs.1"],
        "prior_instance_branch.2.inst_convs.3":  ["prior_instance_branch.2.0.inst_convs.3", "prior_occluder_branch.2.0.inst_convs.3"],
        "prior_instance_branch.2.inst_convs.4":  ["prior_instance_branch.2.0.inst_convs.4", "prior_occluder_branch.2.0.inst_convs.4"],
        "prior_instance_branch.2.inst_convs.6":  ["prior_instance_branch.2.1.inst_convs.0", "prior_occluder_branch.2.1.inst_convs.0"],
        "prior_instance_branch.2.inst_convs.7":  ["prior_instance_branch.2.1.inst_convs.1", "prior_occluder_branch.2.1.inst_convs.1"],
        "prior_instance_branch.2.inst_convs.9":  ["prior_instance_branch.2.1.inst_convs.3", "prior_occluder_branch.2.1.inst_convs.3"],
        "prior_instance_branch.2.inst_convs.10": ["prior_instance_branch.2.1.inst_convs.4", "prior_occluder_branch.2.1.inst_convs.4"],

        "prior_instance_branch.2.inst_convs.12":  ["prior_instance_branch.2.2.inst_convs.0", "prior_occluder_branch.2.2.inst_convs.0"],
        "prior_instance_branch.2.inst_convs.13":  ["prior_instance_branch.2.2.inst_convs.1", "prior_occluder_branch.2.2.inst_convs.1"],
        "prior_instance_branch.2.inst_convs.15":  ["prior_instance_branch.2.2.inst_convs.3", "prior_occluder_branch.2.2.inst_convs.3"],
        "prior_instance_branch.2.inst_convs.16":  ["prior_instance_branch.2.2.inst_convs.4", "prior_occluder_branch.2.2.inst_convs.4"],


        "prior_instance_branch.3.inst_convs.0":  ["prior_instance_branch.3.0.inst_convs.0", "prior_occluder_branch.3.0.inst_convs.0"],
        "prior_instance_branch.3.inst_convs.1":  ["prior_instance_branch.3.0.inst_convs.1", "prior_occluder_branch.3.0.inst_convs.1"],
        "prior_instance_branch.3.inst_convs.3":  ["prior_instance_branch.3.0.inst_convs.3", "prior_occluder_branch.3.0.inst_convs.3"],
        "prior_instance_branch.3.inst_convs.4":  ["prior_instance_branch.3.0.inst_convs.4", "prior_occluder_branch.3.0.inst_convs.4"],
        "prior_instance_branch.3.inst_convs.6":  ["prior_instance_branch.3.1.inst_convs.0", "prior_occluder_branch.3.1.inst_convs.0"],
        "prior_instance_branch.3.inst_convs.7":  ["prior_instance_branch.3.1.inst_convs.1", "prior_occluder_branch.3.1.inst_convs.1"],
        "prior_instance_branch.3.inst_convs.9":  ["prior_instance_branch.3.1.inst_convs.3", "prior_occluder_branch.3.1.inst_convs.3"],
        "prior_instance_branch.3.inst_convs.10": ["prior_instance_branch.3.1.inst_convs.4", "prior_occluder_branch.3.1.inst_convs.4"],
        
        "prior_instance_branch.3.inst_convs.12":  ["prior_instance_branch.3.2.inst_convs.0", "prior_occluder_branch.3.2.inst_convs.0"],
        "prior_instance_branch.3.inst_convs.13":  ["prior_instance_branch.3.2.inst_convs.1", "prior_occluder_branch.3.2.inst_convs.1"],
        "prior_instance_branch.3.inst_convs.15":  ["prior_instance_branch.3.2.inst_convs.3", "prior_occluder_branch.3.2.inst_convs.3"],
        "prior_instance_branch.3.inst_convs.16":  ["prior_instance_branch.3.2.inst_convs.4", "prior_occluder_branch.3.2.inst_convs.4"],


        "prior_instance_branch.4.inst_convs.0":  ["prior_instance_branch.4.0.inst_convs.0", "prior_occluder_branch.4.0.inst_convs.0"],
        "prior_instance_branch.4.inst_convs.1":  ["prior_instance_branch.4.0.inst_convs.1", "prior_occluder_branch.4.0.inst_convs.1"],
        "prior_instance_branch.4.inst_convs.3":  ["prior_instance_branch.4.0.inst_convs.3", "prior_occluder_branch.4.0.inst_convs.3"],
        "prior_instance_branch.4.inst_convs.4":  ["prior_instance_branch.4.0.inst_convs.4", "prior_occluder_branch.4.0.inst_convs.4"],
        "prior_instance_branch.4.inst_convs.6":  ["prior_instance_branch.4.1.inst_convs.0", "prior_occluder_branch.4.1.inst_convs.0"],
        "prior_instance_branch.4.inst_convs.7":  ["prior_instance_branch.4.1.inst_convs.1", "prior_occluder_branch.4.1.inst_convs.1"],
        "prior_instance_branch.4.inst_convs.9":  ["prior_instance_branch.4.1.inst_convs.3", "prior_occluder_branch.4.1.inst_convs.3"],
        "prior_instance_branch.4.inst_convs.10": ["prior_instance_branch.4.1.inst_convs.4", "prior_occluder_branch.4.1.inst_convs.4"],
        
        "prior_instance_branch.4.inst_convs.12":  ["prior_instance_branch.4.2.inst_convs.0", "prior_occluder_branch.4.2.inst_convs.0"],
        "prior_instance_branch.4.inst_convs.13":  ["prior_instance_branch.4.2.inst_convs.1", "prior_occluder_branch.4.2.inst_convs.1"],
        "prior_instance_branch.4.inst_convs.15":  ["prior_instance_branch.4.2.inst_convs.3", "prior_occluder_branch.4.2.inst_convs.3"],
        "prior_instance_branch.4.inst_convs.16":  ["prior_instance_branch.4.2.inst_convs.4", "prior_occluder_branch.4.2.inst_convs.4"],
    }

    # module_mapping.update(prior_instance_branch_mappings)
    
    # module_mapping = {
    #     "up_se_blocks": ["up_se_blocks_occluders"],
    #     "up_conv_layers": ["up_conv_layers_occluders"],
    #     "mask_branch": ["mask_branch_occluder", "out_mask_branch_occluder", "out_mask_branch"],
    #     # "prior_instance_branch": ["prior_occluder_branch", "out_instance_branch", "out_occluder_branch"],
    #     "prior_instance_branch.0": ["prior_instance_branch.0"
    #                                 "prior_occluder_branch.0"],
    #     "prior_instance_branch.1": ["prior_instance_branch.0", "prior_instance_branch.1", "prior_instance_branch.2", "prior_instance_branch.3", "prior_instance_branch.4",
    #                                 "prior_occluder_branch.0", "prior_occluder_branch.1", "prior_occluder_branch.2", "prior_occluder_branch.3", "prior_occluder_branch.4"],
    #     "instance_branch": ["occluder_branch"],

    #     # "up_se_blocks": "up_se_block_overlaps",
    #     # "up_conv_layers": "up_conv_layers_overlaps",
    #     # "mask_branch": "mask_branch_overlap",
    #     # "prior_instance_branch": "prior_overlap_branch",
    #     # "instance_branch": "overlap_branch"
    # }

    for k, v in loaded_state_dict.items():
        # Load weights for modules in module_mapping
        for key, values in module_mapping.items():
            if k.startswith(key):
                if k in current_model_dict and v.size() == current_model_dict[k].size():
                    print(f'Loading original value {k}')
                    print(k in current_model_dict)
                    current_model_dict[k] = v
                # elif k in current_model_dict:
                #     print(f"WARNING: Skipping loading weights for parameter '{k}' due to size mismatch.")
                #     print(f"Expected size: {current_model_dict[k].size()}, but got size: {v.size()}")
                # else:
                #     print(f"WARNING: Skipping loading weights for parameter '{k}' as it was not found in the current model.")
            
                # for value in values:
                #     mapped_name = k.replace(key, value)
                #     print(f'{mapped_name in current_model_dict}: Mapping {k} to {mapped_name}')
                #     if mapped_name in current_model_dict and v.size() == current_model_dict[mapped_name].size():
                #         current_model_dict[mapped_name] = v

                    # elif mapped_name in current_model_dict:
                    #     print(f"WARNING: Skipping loading weights for parameter '{mapped_name}' due to size mismatch.")
                    #     print(f"Expected size: {current_model_dict[mapped_name].size()}, but got size: {v.size()}")
                    # else:
                    #     print(f"WARNING: Skipping loading weights for parameter '{mapped_name}' as it was not found in the current model.")
            

        # Load weights for all other modules
        # if k in current_model_dict and v.size() == current_model_dict[k].size():
        #     current_model_dict[k] = v
        # elif k in current_model_dict:
        #     print(f"WARNING: Skipping loading weights for parameter '{k}' due to size mismatch.")
        #     print(f"Expected size: {current_model_dict[k].size()}, but got size: {v.size()}")
        # else:
        #     print(f"WARNING: Skipping loading weights for parameter '{k}' as it was not found in the current model.")
    


    # for k, v in loaded_state_dict.items():
    #     if k in current_model_dict and v.size() == current_model_dict[k].size():
    #         current_model_dict[k] = v
    #     elif k in current_model_dict:
    #         print(f"WARNING: Skipping loading weights for parameter '{k}' due to size mismatch.")
    #         print(f"Expected size: {current_model_dict[k].size()}, but got size: {v.size()}")
    #     else:
    #         print(f"WARNING: Skipping loading weights for parameter '{k}' as it was not found in the current model.")
    
    # # Warn about weights in the current model but not present in the pretrained weights
    # for k in current_model_dict.keys():
    #     if k not in loaded_state_dict:
            # print(f"WARNING: Parameter '{k}' in the current model is not present in the pretrained weights.")

    model.load_state_dict(current_model_dict, strict=False)

    print("- Weights loaded!")
    
    return model


def get_model_from_path(cfg: cfg):
    # runs/.../sparse_seunet.py
    
    # model_file = f"{cfg.model.model_files}/models/{cfg.model.arch}.py"
    # model_file = f"{cfg.model.model_files}/models/__init__.py"
    model_file = f"{cfg.model.model_files}/__init__.py"
    assert isfile(model_file), FileNotFoundError(f"Model file not found: {model_file}")

    if cfg.verbose: 
        print("Loading model from path...")
        print(f"Found model files: "
              f"\n- {cfg.model.model_files}")

    # TODO: pass imports to change
    # modify_import_statements(cfg, model_files=cfg.model.model_files)
    
    module = import_from_file(model_file, clear_cache=True)
    model_class = getattr(module, 'SparseSEUnet')
    model = model_class(cfg)

    return model


def _copy_folder(src, dst, ignore=None):
    source_parent_folder = os.path.basename(src)
    destination_parent = os.path.join(dst, source_parent_folder)
    shutil.copytree(src, destination_parent, ignore=ignore)


# NOTE: should be the same as src directory structure
def save_model_files(arch, save_dir):
    dst = save_dir / 'model_files'
    makedirs(dst, exist_ok=True)
    # shutil.copy(join(MODEL_FILES, 'models', f'{arch}.py'), save_dir / 'model_files')
    # shutil.copytree(join(MODEL_FILES, 'heads/instance_head/'), save_dir / 'model_files')

    for src in ["__init__.py", "matcher.py", "loss.py"]:
        shutil.copy(
            join(MODEL_FILES, src),
            dst
            )
    _copy_folder(
        src=join(MODEL_FILES, 'heads'),
        dst=dst
        )
    
    keep = [arch, '__init__']
    _copy_folder(
        src=join(MODEL_FILES, 'models'),
        dst=dst,
        ignore=lambda src, names: [name for name in names if all(k not in name for k in keep)]
        )
    
    # shutil.copy(
    #     join(MODEL_FILES, '__init__.py'),
    #     dst
    #     )
    
    # change init model imports
    init_file = join(dst, 'models', '__init__.py')

    with open(init_file, 'r') as f:
        lines = f.readlines()

    # from .sparse_seunet_add_overlaps import SparseSEUnet as SparseSEUnetAddOverlaps ->
    # -> from .sparse_seunet_add_overlaps import SparseSEUnet
    # keep = arch
    # filtered_lines = [
    #     re.sub(rf'from .* import {keep} as .*', f'from .* import {keep}', line)
    #     if re.search(rf'from .* import {keep}(?!\w)', line) else line
    #     for line in lines
    # ]

    # Create a regular expression pattern for the import statement
    pattern = rf'from \..* import {arch}(?!\w) as {arch}'

    # Filter out lines that match the pattern and remove the renaming part
    filtered_lines = [
        re.sub(pattern, f'from .{arch} import {arch}', line)
        if re.search(pattern, line)
        else line
        for line in lines
    ]

    with open(init_file, 'w') as f:
        f.writelines(filtered_lines)






    


    

# prior_instance_branch_mappings = {
    #     "prior_instance_branch.0.inst_convs.0":  ["prior_instance_branch.0.0.inst_convs.0", "prior_occluder_branch.0.0.inst_convs.0"],
    #     "prior_instance_branch.0.inst_convs.1":  ["prior_instance_branch.0.0.inst_convs.1", "prior_occluder_branch.0.0.inst_convs.1"],
    #     "prior_instance_branch.0.inst_convs.3":  ["prior_instance_branch.0.0.inst_convs.3", "prior_occluder_branch.0.0.inst_convs.3"],
    #     "prior_instance_branch.0.inst_convs.4":  ["prior_instance_branch.0.0.inst_convs.4", "prior_occluder_branch.0.0.inst_convs.4"],
    #     "prior_instance_branch.0.inst_convs.6":  ["prior_instance_branch.0.1.inst_convs.0", "prior_occluder_branch.0.1.inst_convs.0"],
    #     "prior_instance_branch.0.inst_convs.7":  ["prior_instance_branch.0.1.inst_convs.1", "prior_occluder_branch.0.1.inst_convs.1"],
    #     "prior_instance_branch.0.inst_convs.9":  ["prior_instance_branch.0.1.inst_convs.3", "prior_occluder_branch.0.1.inst_convs.3"],
    #     "prior_instance_branch.0.inst_convs.10": ["prior_instance_branch.0.1.inst_convs.4", "prior_occluder_branch.0.1.inst_convs.4"],
 
    #     "prior_instance_branch.1.inst_convs.0":  ["prior_instance_branch.1.0.inst_convs.0", "prior_occluder_branch.1.0.inst_convs.0"],
    #     "prior_instance_branch.1.inst_convs.1":  ["prior_instance_branch.1.0.inst_convs.1", "prior_occluder_branch.1.0.inst_convs.1"],
    #     "prior_instance_branch.1.inst_convs.3":  ["prior_instance_branch.1.0.inst_convs.3", "prior_occluder_branch.1.0.inst_convs.3"],
    #     "prior_instance_branch.1.inst_convs.4":  ["prior_instance_branch.1.0.inst_convs.4", "prior_occluder_branch.1.0.inst_convs.4"],
    #     "prior_instance_branch.1.inst_convs.6":  ["prior_instance_branch.1.1.inst_convs.0", "prior_occluder_branch.1.1.inst_convs.0"],
    #     "prior_instance_branch.1.inst_convs.7":  ["prior_instance_branch.1.1.inst_convs.1", "prior_occluder_branch.1.1.inst_convs.1"],
    #     "prior_instance_branch.1.inst_convs.9":  ["prior_instance_branch.1.1.inst_convs.3", "prior_occluder_branch.1.1.inst_convs.3"],
    #     "prior_instance_branch.1.inst_convs.10": ["prior_instance_branch.1.1.inst_convs.4", "prior_occluder_branch.1.1.inst_convs.4"],

    #     "prior_instance_branch.2.inst_convs.0":  ["prior_instance_branch.2.0.inst_convs.0", "prior_occluder_branch.2.0.inst_convs.0"],
    #     "prior_instance_branch.2.inst_convs.1":  ["prior_instance_branch.2.0.inst_convs.1", "prior_occluder_branch.2.0.inst_convs.1"],
    #     "prior_instance_branch.2.inst_convs.3":  ["prior_instance_branch.2.0.inst_convs.3", "prior_occluder_branch.2.0.inst_convs.3"],
    #     "prior_instance_branch.2.inst_convs.4":  ["prior_instance_branch.2.0.inst_convs.4", "prior_occluder_branch.2.0.inst_convs.4"],
    #     "prior_instance_branch.2.inst_convs.6":  ["prior_instance_branch.2.1.inst_convs.0", "prior_occluder_branch.2.1.inst_convs.0"],
    #     "prior_instance_branch.2.inst_convs.7":  ["prior_instance_branch.2.1.inst_convs.1", "prior_occluder_branch.2.1.inst_convs.1"],
    #     "prior_instance_branch.2.inst_convs.9":  ["prior_instance_branch.2.1.inst_convs.3", "prior_occluder_branch.2.1.inst_convs.3"],
    #     "prior_instance_branch.2.inst_convs.10": ["prior_instance_branch.2.1.inst_convs.4", "prior_occluder_branch.2.1.inst_convs.4"],

    #     "prior_instance_branch.3.inst_convs.0":  ["prior_instance_branch.3.0.inst_convs.0", "prior_occluder_branch.3.0.inst_convs.0"],
    #     "prior_instance_branch.3.inst_convs.1":  ["prior_instance_branch.3.0.inst_convs.1", "prior_occluder_branch.3.0.inst_convs.1"],
    #     "prior_instance_branch.3.inst_convs.3":  ["prior_instance_branch.3.0.inst_convs.3", "prior_occluder_branch.3.0.inst_convs.3"],
    #     "prior_instance_branch.3.inst_convs.4":  ["prior_instance_branch.3.0.inst_convs.4", "prior_occluder_branch.3.0.inst_convs.4"],
    #     "prior_instance_branch.3.inst_convs.6":  ["prior_instance_branch.3.1.inst_convs.0", "prior_occluder_branch.3.1.inst_convs.0"],
    #     "prior_instance_branch.3.inst_convs.7":  ["prior_instance_branch.3.1.inst_convs.1", "prior_occluder_branch.3.1.inst_convs.1"],
    #     "prior_instance_branch.3.inst_convs.9":  ["prior_instance_branch.3.1.inst_convs.3", "prior_occluder_branch.3.1.inst_convs.3"],
    #     "prior_instance_branch.3.inst_convs.10": ["prior_instance_branch.3.1.inst_convs.4", "prior_occluder_branch.3.1.inst_convs.4"],

    #     "prior_instance_branch.4.inst_convs.0":  ["prior_instance_branch.4.0.inst_convs.0", "prior_occluder_branch.4.0.inst_convs.0"],
    #     "prior_instance_branch.4.inst_convs.1":  ["prior_instance_branch.4.0.inst_convs.1", "prior_occluder_branch.4.0.inst_convs.1"],
    #     "prior_instance_branch.4.inst_convs.3":  ["prior_instance_branch.4.0.inst_convs.3", "prior_occluder_branch.4.0.inst_convs.3"],
    #     "prior_instance_branch.4.inst_convs.4":  ["prior_instance_branch.4.0.inst_convs.4", "prior_occluder_branch.4.0.inst_convs.4"],
    #     "prior_instance_branch.4.inst_convs.6":  ["prior_instance_branch.4.1.inst_convs.0", "prior_occluder_branch.4.1.inst_convs.0"],
    #     "prior_instance_branch.4.inst_convs.7":  ["prior_instance_branch.4.1.inst_convs.1", "prior_occluder_branch.4.1.inst_convs.1"],
    #     "prior_instance_branch.4.inst_convs.9":  ["prior_instance_branch.4.1.inst_convs.3", "prior_occluder_branch.4.1.inst_convs.3"],
    #     "prior_instance_branch.4.inst_convs.10": ["prior_instance_branch.4.1.inst_convs.4", "prior_occluder_branch.4.1.inst_convs.4"],
    # }


    # prior_instance_branch_mappings = {
    #     "prior_instance_branch.0.inst_convs.0":  ["prior_instance_branch.0.0.inst_convs.0", "prior_occluder_branch.0.0.inst_convs.0"],
    #     "prior_instance_branch.0.inst_convs.1":  ["prior_instance_branch.0.0.inst_convs.1", "prior_occluder_branch.0.0.inst_convs.1"],
    #     "prior_instance_branch.0.inst_convs.3":  ["prior_instance_branch.0.0.inst_convs.3", "prior_occluder_branch.0.0.inst_convs.3"],
    #     "prior_instance_branch.0.inst_convs.4":  ["prior_instance_branch.0.0.inst_convs.4", "prior_occluder_branch.0.0.inst_convs.4"],
    #     "prior_instance_branch.0.inst_convs.6":  ["prior_instance_branch.0.1.inst_convs.0", "prior_occluder_branch.0.1.inst_convs.0"],
    #     "prior_instance_branch.0.inst_convs.7":  ["prior_instance_branch.0.1.inst_convs.1", "prior_occluder_branch.0.1.inst_convs.1"],
    #     "prior_instance_branch.0.inst_convs.9":  ["prior_instance_branch.0.1.inst_convs.3", "prior_occluder_branch.0.1.inst_convs.3"],
    #     "prior_instance_branch.0.inst_convs.10": ["prior_instance_branch.0.1.inst_convs.4", "prior_occluder_branch.0.1.inst_convs.4"],
        
    #     "prior_instance_branch.0.inst_convs.6":  ["prior_instance_branch.0.2.inst_convs.0", "prior_occluder_branch.0.2.inst_convs.0"],
    #     "prior_instance_branch.0.inst_convs.7":  ["prior_instance_branch.0.2.inst_convs.1", "prior_occluder_branch.0.2.inst_convs.1"],
    #     "prior_instance_branch.0.inst_convs.9":  ["prior_instance_branch.0.2.inst_convs.3", "prior_occluder_branch.0.2.inst_convs.3"],
    #     "prior_instance_branch.0.inst_convs.10": ["prior_instance_branch.0.2.inst_convs.4", "prior_occluder_branch.0.2.inst_convs.4"],
 

    #     "prior_instance_branch.1.inst_convs.0":  ["prior_instance_branch.1.0.inst_convs.0", "prior_occluder_branch.1.0.inst_convs.0"],
    #     "prior_instance_branch.1.inst_convs.1":  ["prior_instance_branch.1.0.inst_convs.1", "prior_occluder_branch.1.0.inst_convs.1"],
    #     "prior_instance_branch.1.inst_convs.3":  ["prior_instance_branch.1.0.inst_convs.3", "prior_occluder_branch.1.0.inst_convs.3"],
    #     "prior_instance_branch.1.inst_convs.4":  ["prior_instance_branch.1.0.inst_convs.4", "prior_occluder_branch.1.0.inst_convs.4"],
    #     "prior_instance_branch.1.inst_convs.6":  ["prior_instance_branch.1.1.inst_convs.0", "prior_occluder_branch.1.1.inst_convs.0"],
    #     "prior_instance_branch.1.inst_convs.7":  ["prior_instance_branch.1.1.inst_convs.1", "prior_occluder_branch.1.1.inst_convs.1"],
    #     "prior_instance_branch.1.inst_convs.9":  ["prior_instance_branch.1.1.inst_convs.3", "prior_occluder_branch.1.1.inst_convs.3"],
    #     "prior_instance_branch.1.inst_convs.10": ["prior_instance_branch.1.1.inst_convs.4", "prior_occluder_branch.1.1.inst_convs.4"],
        
    #     "prior_instance_branch.1.inst_convs.6":  ["prior_instance_branch.1.2.inst_convs.0", "prior_occluder_branch.1.2.inst_convs.0"],
    #     "prior_instance_branch.1.inst_convs.7":  ["prior_instance_branch.1.2.inst_convs.1", "prior_occluder_branch.1.2.inst_convs.1"],
    #     "prior_instance_branch.1.inst_convs.9":  ["prior_instance_branch.1.2.inst_convs.3", "prior_occluder_branch.1.2.inst_convs.3"],
    #     "prior_instance_branch.1.inst_convs.10": ["prior_instance_branch.1.2.inst_convs.4", "prior_occluder_branch.1.2.inst_convs.4"],


    #     "prior_instance_branch.2.inst_convs.0":  ["prior_instance_branch.2.0.inst_convs.0", "prior_occluder_branch.2.0.inst_convs.0"],
    #     "prior_instance_branch.2.inst_convs.1":  ["prior_instance_branch.2.0.inst_convs.1", "prior_occluder_branch.2.0.inst_convs.1"],
    #     "prior_instance_branch.2.inst_convs.3":  ["prior_instance_branch.2.0.inst_convs.3", "prior_occluder_branch.2.0.inst_convs.3"],
    #     "prior_instance_branch.2.inst_convs.4":  ["prior_instance_branch.2.0.inst_convs.4", "prior_occluder_branch.2.0.inst_convs.4"],
    #     "prior_instance_branch.2.inst_convs.6":  ["prior_instance_branch.2.1.inst_convs.0", "prior_occluder_branch.2.1.inst_convs.0"],
    #     "prior_instance_branch.2.inst_convs.7":  ["prior_instance_branch.2.1.inst_convs.1", "prior_occluder_branch.2.1.inst_convs.1"],
    #     "prior_instance_branch.2.inst_convs.9":  ["prior_instance_branch.2.1.inst_convs.3", "prior_occluder_branch.2.1.inst_convs.3"],
    #     "prior_instance_branch.2.inst_convs.10": ["prior_instance_branch.2.1.inst_convs.4", "prior_occluder_branch.2.1.inst_convs.4"],

    #     "prior_instance_branch.2.inst_convs.6":  ["prior_instance_branch.2.2.inst_convs.0", "prior_occluder_branch.2.2.inst_convs.0"],
    #     "prior_instance_branch.2.inst_convs.7":  ["prior_instance_branch.2.2.inst_convs.1", "prior_occluder_branch.2.2.inst_convs.1"],
    #     "prior_instance_branch.2.inst_convs.9":  ["prior_instance_branch.2.2.inst_convs.3", "prior_occluder_branch.2.2.inst_convs.3"],
    #     "prior_instance_branch.2.inst_convs.10": ["prior_instance_branch.2.2.inst_convs.4", "prior_occluder_branch.2.2.inst_convs.4"],


    #     "prior_instance_branch.3.inst_convs.0":  ["prior_instance_branch.3.0.inst_convs.0", "prior_occluder_branch.3.0.inst_convs.0"],
    #     "prior_instance_branch.3.inst_convs.1":  ["prior_instance_branch.3.0.inst_convs.1", "prior_occluder_branch.3.0.inst_convs.1"],
    #     "prior_instance_branch.3.inst_convs.3":  ["prior_instance_branch.3.0.inst_convs.3", "prior_occluder_branch.3.0.inst_convs.3"],
    #     "prior_instance_branch.3.inst_convs.4":  ["prior_instance_branch.3.0.inst_convs.4", "prior_occluder_branch.3.0.inst_convs.4"],
    #     "prior_instance_branch.3.inst_convs.6":  ["prior_instance_branch.3.1.inst_convs.0", "prior_occluder_branch.3.1.inst_convs.0"],
    #     "prior_instance_branch.3.inst_convs.7":  ["prior_instance_branch.3.1.inst_convs.1", "prior_occluder_branch.3.1.inst_convs.1"],
    #     "prior_instance_branch.3.inst_convs.9":  ["prior_instance_branch.3.1.inst_convs.3", "prior_occluder_branch.3.1.inst_convs.3"],
    #     "prior_instance_branch.3.inst_convs.10": ["prior_instance_branch.3.1.inst_convs.4", "prior_occluder_branch.3.1.inst_convs.4"],
        
    #     "prior_instance_branch.3.inst_convs.6":  ["prior_instance_branch.3.2.inst_convs.0", "prior_occluder_branch.3.2.inst_convs.0"],
    #     "prior_instance_branch.3.inst_convs.7":  ["prior_instance_branch.3.2.inst_convs.1", "prior_occluder_branch.3.2.inst_convs.1"],
    #     "prior_instance_branch.3.inst_convs.9":  ["prior_instance_branch.3.2.inst_convs.3", "prior_occluder_branch.3.2.inst_convs.3"],
    #     "prior_instance_branch.3.inst_convs.10": ["prior_instance_branch.3.2.inst_convs.4", "prior_occluder_branch.3.2.inst_convs.4"],


    #     "prior_instance_branch.4.inst_convs.0":  ["prior_instance_branch.4.0.inst_convs.0", "prior_occluder_branch.4.0.inst_convs.0"],
    #     "prior_instance_branch.4.inst_convs.1":  ["prior_instance_branch.4.0.inst_convs.1", "prior_occluder_branch.4.0.inst_convs.1"],
    #     "prior_instance_branch.4.inst_convs.3":  ["prior_instance_branch.4.0.inst_convs.3", "prior_occluder_branch.4.0.inst_convs.3"],
    #     "prior_instance_branch.4.inst_convs.4":  ["prior_instance_branch.4.0.inst_convs.4", "prior_occluder_branch.4.0.inst_convs.4"],
    #     "prior_instance_branch.4.inst_convs.6":  ["prior_instance_branch.4.1.inst_convs.0", "prior_occluder_branch.4.1.inst_convs.0"],
    #     "prior_instance_branch.4.inst_convs.7":  ["prior_instance_branch.4.1.inst_convs.1", "prior_occluder_branch.4.1.inst_convs.1"],
    #     "prior_instance_branch.4.inst_convs.9":  ["prior_instance_branch.4.1.inst_convs.3", "prior_occluder_branch.4.1.inst_convs.3"],
    #     "prior_instance_branch.4.inst_convs.10": ["prior_instance_branch.4.1.inst_convs.4", "prior_occluder_branch.4.1.inst_convs.4"],
        
    #     "prior_instance_branch.4.inst_convs.6":  ["prior_instance_branch.4.2.inst_convs.0", "prior_occluder_branch.4.2.inst_convs.0"],
    #     "prior_instance_branch.4.inst_convs.7":  ["prior_instance_branch.4.2.inst_convs.1", "prior_occluder_branch.4.2.inst_convs.1"],
    #     "prior_instance_branch.4.inst_convs.9":  ["prior_instance_branch.4.2.inst_convs.3", "prior_occluder_branch.4.2.inst_convs.3"],
    #     "prior_instance_branch.4.inst_convs.10": ["prior_instance_branch.4.2.inst_convs.4", "prior_occluder_branch.4.2.inst_convs.4"],
    # }



# NOTE: needed to refactor older version file structure in model_files -- done
def modify_import_statements(cfg: cfg, model_files):
    """
    Dynamically load model files
    """
    model_file = f'{model_files}/{cfg.model.arch}.py'

    # Read the content of the model.py file
    with open(model_file, 'r') as f:
        model_content = f.read()

    # Modify the import statements
    # model_content = model_content.replace(
    #     f"{MODEL_FILES}/heads/instance_head/instance_head.py",
    #     f"{model_files}/instance_head.py"
    # )

    # model_content = model_content.replace(
    #     "from models.seg.heads.instance_head import InstanceBranch, PriorInstanceBranch, GroupInstanceBranch",
    #     '# Specify the path to instance_head.py\n'
    #     f'instance_head_file = "{model_files}/instance_head.py"\n\n'
    #     '# Load the instance_head.py file as a module\n'
    #     'spec = importlib.util.spec_from_file_location("instance_head", instance_head_file)\n'
    #     'instance_head = importlib.util.module_from_spec(spec)\n'
    #     'spec.loader.exec_module(instance_head)\n\n'
    #     'InstanceBranch = instance_head.InstanceBranch\n'
    #     'PriorInstanceBranch = instance_head.PriorInstanceBranch\n'
    #     'GroupInstanceBranch = instance_head.GroupInstanceBranch'
    # )

    # if cfg.verbose: 
    #     print(f"Modified imports: "
    #           f"\n- {model_file} "
    #           f"\n[{MODEL_FILES}/heads/instance_head/instance_head.py] -> [{model_files}/instance_head.py]")

    # Create a temporary modified file
    with open(model_file, 'w') as f:
        f.write(model_content)