from torch import nn
from fvcore.nn.weight_init import c2_msra_fill
from utils.registry import HEADS


def _make_stack_3x3_convs(num_convs, in_channels, out_channels):
    convs = []
    for _ in range(num_convs):
        convs.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        convs.append(nn.BatchNorm2d(out_channels))
        convs.append(nn.ReLU(inplace=True))

        in_channels = out_channels
    return nn.Sequential(*convs)


@HEADS.register(name="MaskStackedConv")
class MaskBranch(nn.Module):
    def __init__(self, in_channels, out_channels=256, num_convs=4):
        super().__init__()
        self.mask_convs = _make_stack_3x3_convs(num_convs, in_channels, out_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.mask_convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)

    def forward(self, features):
        features = self.mask_convs(features)
        return features

