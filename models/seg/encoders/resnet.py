import torch
import torchvision.models as models
from torch import nn

import sys
sys.path.append("./")

from utils.registry import MODELS


@MODELS.register(name="ResNet")
class ResNet(nn.Module):
    def __init__(self, 
        depth=50,
        num_stages=5,
        out_indices=(0, 1, 2, 3, 4),
        pretrained=True
        ):
        super().__init__()
        depth = depth
        self.num_stages = num_stages
        self.out_indices = out_indices
        assert self.num_stages <= len(self.out_indices)

        if depth == 34:
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            model = models.resnet34(weights=weights)
            self.embed_dims = [64, 64, 128, 256, 512]
        elif depth == 50:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            self.embed_dims = [64, 256, 512, 1024, 2048]
        elif depth == 101:
            weights = models.ResNet101_Weights.DEFAULT if pretrained else None
            model = models.resnet101(weights=weights)
            self.embed_dims = [64, 256, 512, 1024, 2048]
        elif depth == 152:
            weights = models.ResNet152_Weights.DEFAULT if pretrained else None
            model = models.resnet152(weights=weights)
            self.embed_dims = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")

        self.embed_dims = [self.embed_dims[i] for i in out_indices]

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

        self.res_layers = []
        stage_names = ['layer1', 'layer2', 'layer3', 'layer4']
        for i in range(self.num_stages):
            layer_name = stage_names[i]
            res_layer = getattr(model, layer_name)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

    def forward(self, x):
        outputs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if 0 in self.out_indices:
            outputs.append(x)
            
        x = self.maxpool(x)

        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i + 1 in self.out_indices:
                outputs.append(x)

        return outputs

