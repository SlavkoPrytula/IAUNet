from os.path import join, dirname, isfile
from os import makedirs
import shutil
import importlib.util
import re
import sys

from types import ModuleType
from typing import Type, cast
from typing_extensions import Protocol

import torch

from models.seg import SparseSEUnet
from models.seg import SparseSEUnetAddOverlaps
# from models.seg.models.sparse_seunet_cat_overlaps import SparseSEUnet as SparseSEUnetCatOverlaps
# from models.seg.models.sparse_seunet_kernel_fusion import SparseSEUnet as SparseSEUnetKernelFusion
# from models.seg.models.sparse_seunet_feat_iam_mix import SparseSEUnet as SparseSEUnetFeatIAMMix

# from models.seg.models.sparse_seunet_ovlp_single_branch import SparseSEUnet as SparseSEUnet_OverlapsSingleBranch
# from models.seg.models.sparse_seunet_ovlp_cat_v0 import SparseSEUnet as SparseSEUnet_OverlapsCat_v0
# from models.seg.models.sparse_seunet_ovlp_attn_v0 import SparseSEUnet as SparseSEUnet_OverlapsAttn_v0

from configs import cfg


MODEL_FILES = "/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/models/seg"


patterns = [
    r'from models\.seg\.heads\.instance_head\b.*'
    ]

mapping = ['models/seg/heads/instance_head/instance_head.py']


def import_from_file(module_name, file_path) -> Type[ModuleType]:
    """
    Dynamically load module from file

    :param source: file to load
    :param module_name: name of module to register in sys.modules
    :return: loaded module
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module



# NOTE: needed to older version file structure in model_files
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
    
    module = import_from_file('SparseSEUnet', model_file)
    model_class = getattr(module, 'SparseSEUnet')
    model = model_class(cfg)

    return model


def get_model(cfg: cfg):
    # TODO: rewrite files loading for models
    if cfg.model.load_from_files:
        # If model_file is specified, load the model from the file
        model = get_model_from_path(cfg)
    else:
        models = {
            'sparse_seunet': SparseSEUnet,
            # 'sparse_seunet_kernel_fusion': SparseSEUnetKernelFusion,
            # 'sparse_seunet_feat_iam_mix': SparseSEUnetFeatIAMMix,
            'sparse_seunet_add_overlaps': SparseSEUnetAddOverlaps,
            # 'sparse_seunet_cat_overlaps': SparseSEUnetCatOverlaps
            # 'sparse_seunet_ovlp_single_branch': SparseSEUnet_OverlapsSingleBranch,
            # 'sparse_seunet_ovlp_cat_v0': SparseSEUnet_OverlapsCat_v0,
            # 'sparse_seunet_ovlp_attn_v0': SparseSEUnet_OverlapsAttn_v0
        }

        model = models[cfg.model.arch](
            cfg=cfg,
            )
    
    return model


def load_weights(model, weights_path):
    print(f"- Loading pretrained weights:\n[{weights_path}]")

    current_model_dict = model.state_dict()
    loaded_state_dict = torch.load(weights_path)

    for k, v in loaded_state_dict.items():
        if k in current_model_dict and v.size() == current_model_dict[k].size():
            current_model_dict[k] = v
        elif k in current_model_dict:
            print(f"WARNING: Skipping loading weights for parameter '{k}' due to size mismatch.")
            print(f"Expected size: {current_model_dict[k].size()}, but got size: {v.size()}")
        else:
            print(f"WARNING: Skipping loading weights for parameter '{k}' as it was not found in the current model.")
    
    # Warn about weights in the current model but not present in the pretrained weights
    for k in current_model_dict.keys():
        if k not in loaded_state_dict:
            print(f"WARNING: Parameter '{k}' in the current model is not present in the pretrained weights.")

    model.load_state_dict(current_model_dict, strict=False)

    print("- Weights loaded!")
    
    return model


def save_model_files(arch, save_dir):
    makedirs(save_dir / 'model_files', exist_ok=True)
    shutil.copy(join(MODEL_FILES, 'models', f'{arch}.py'), save_dir / 'model_files')
    shutil.copytree(join(MODEL_FILES, 'heads/instance_head/'), save_dir / 'model_files')