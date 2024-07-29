import contextlib
import glob
import os
import urllib
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile
import shutil


def increment_path(path, exist_ok=False, sep='_', mkdir=False):
    """
    Increments a file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    If the path exists and exist_ok is not set to True, the path will be incremented by appending a number and sep to
    the end of the path. If the path is a file, the file extension will be preserved. If the path is a directory, the
    number will be appended directly to the end of the path. If mkdir is set to True, the path will be created as a
    directory if it does not already exist.
    Args:
    path (str or pathlib.Path): Path to increment.
    exist_ok (bool, optional): If True, the path will not be incremented and will be returned as-is. Defaults to False.
    sep (str, optional): Separator to use between the path and the incrementation number. Defaults to an empty string.
    mkdir (bool, optional): If True, the path will be created as a directory if it does not exist. Defaults to False.
    Returns:
    pathlib.Path: Incremented path.
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def _copy_folder(src, dst, base_src_dir=None, ignore=None):
    if base_src_dir is None:
        base_src_dir = os.path.dirname(src)
    
    relative_path = os.path.relpath(src, start=base_src_dir)
    dst_file_path = os.path.join(dst, relative_path)
    os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
    
    if os.path.isdir(src):
        shutil.copytree(src, dst_file_path)
    elif os.path.isfile(src):
        shutil.copy2(src, dst_file_path)
    else:
        raise ValueError(f"Source path '{src}' is not a file or directory")