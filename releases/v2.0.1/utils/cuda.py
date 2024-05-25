import os
from os import mkdir, makedirs
from os.path import join

import torch


def cuda_init(device_id):
    # torch.backends.cudnn.benchmark = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_id}"
    torch.cuda.set_device(device_id)
