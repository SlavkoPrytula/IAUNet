import torch 
from torch import nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

import sys
sys.path.append("./")


class IAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
        super(IAM, self).__init__()
        self.iam_conv = nn.Conv2d(in_channels, out_channels, 
                                  kernel_size=kernel_size, stride=stride, padding=padding, 
                                  groups=groups)
        
        self.prior_prob = 0.01
        self.init_weights()

    def init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        init.normal_(self.iam_conv.weight, std=0.01)
        init.constant_(self.iam_conv.bias, bias_value)

    def forward(self, x):
        x = self.iam_conv(x)
        return x



class IAM_ASPP(nn.Module):
    def __init__(self, in_channels, num_masks, num_groups, base_dilation=1):
        super().__init__()
        self.in_channels = in_channels
        self.num_masks = num_masks
        self.num_groups = num_groups
        self.base_dilation = base_dilation
        
        self.local_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                    padding=1, groups=num_groups)
        
        self.context_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                       padding=self.base_dilation, dilation=self.base_dilation, groups=num_groups)
        self.context_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                       padding=2 * self.base_dilation, dilation=2 * self.base_dilation, groups=num_groups)
        self.context_conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                       padding=4 * self.base_dilation, dilation=4 * self.base_dilation, groups=num_groups)
        
        self.fusion_conv = nn.Conv2d(in_channels * 4, num_masks * num_groups, kernel_size=1, groups=num_groups)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        lf = self.local_conv(x)
        c1 = self.context_conv1(x)
        c2 = self.context_conv2(x)
        c3 = self.context_conv3(x)
        
        concatenated_features = torch.cat([lf, c1, c2, c3], dim=1)
        output_features = self.fusion_conv(concatenated_features)
        
        return output_features





# class IAM(nn.Module):
#     def __init__(self, in_channels, out_channels, groups=1):
#         super(IAM, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)
        
#         self.prior_prob = 0.01
#         self._init_weights()

#     def _init_weights(self):
#         bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)

#         for modules in [self.conv]:
#             for l in modules.modules():
#                 if isinstance(l, nn.Conv2d):
#                     nn.init.normal_(l.weight, std=0.1)
#                     if l.bias is not None:
#                         nn.init.constant_(l.bias, bias_value)

#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class IAM(nn.Module):
#     def __init__(self, in_channels, out_channels, groups=1):
#         super(IAM, self).__init__()
#         self.conv = DynamicConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        
#         self.prior_prob = 0.01
#         self._init_weights()

#     def _init_weights(self):
#         bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)

#         for modules in [self.conv]:
#             for l in modules.modules():
#                 if isinstance(l, nn.Conv2d):
#                     nn.init.normal_(l.weight, std=0.1)
#                     if l.bias is not None:
#                         nn.init.constant_(l.bias, bias_value)

#     def forward(self, x):
#         x = self.conv(x)
        
#         return x


class DeepIAM(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(DeepIAM, self).__init__()
        self.num_convs = 2
        
        convs = []
        for _ in range(self.num_convs):
            convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups))
            convs.append(nn.BatchNorm2d(out_channels))
            convs.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        self.convs = nn.Sequential(*convs)
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)

        self._init_weights()

    def _init_weights(self):
        for m in self.convs.modules():
            if isinstance(m, nn.Conv2d):
                c2_msra_fill(m)
        c2_msra_fill(self.projection)

    def forward(self, x):
        x = self.convs(x)
        x = self.projection(x)
        
        return x


if __name__ == '__main__':
    x = torch.randn(2, 32, 64, 64)
    iam = IAM(32, 10)
    out = iam(x)
    print(out.shape)