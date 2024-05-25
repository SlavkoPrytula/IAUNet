import os
import importlib.util
import shutil

# from .func_1.func_1 import func_1


def _copy_folder(src, dst, ignore=None):
    source_parent_folder = os.path.basename(src)
    destination_parent = os.path.join(dst, source_parent_folder)
    shutil.copytree(src, destination_parent, ignore=ignore)


src = '/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/test/func_1'
dst = '/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/test/func_2'

keep = ['temp', '__init__']
_copy_folder(src, dst, ignore=lambda src, names: [name for name in names if all(k not in name for k in keep)])