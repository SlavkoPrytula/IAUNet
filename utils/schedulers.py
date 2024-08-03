from torch.optim import lr_scheduler
from utils.registry import SCHEDULERS


@SCHEDULERS.register(name="CosineAnnealingLR")
def CosineAnnealingLR():
    return lr_scheduler.CosineAnnealingLR


@SCHEDULERS.register(name="CosineAnnealingWarmRestarts")
def CosineAnnealingWarmRestarts():
    return lr_scheduler.CosineAnnealingWarmRestarts


@SCHEDULERS.register(name="ReduceLROnPlateau")
def ReduceLROnPlateau():
    return lr_scheduler.ReduceLROnPlateau


@SCHEDULERS.register(name="ExponentialLR")
def ExponentialLR():
    return lr_scheduler.ExponentialLR


@SCHEDULERS.register(name="MultiStepLR")
def ExponentialLR():
    return lr_scheduler.MultiStepLR
