import torch 
from torch import digamma, nn
from torch.nn import init
import numpy as np
import torch.nn.functional as F
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

import sys
sys.path.append('.')

from models.seg.heads.instance_head import InstanceHead
from configs import cfg
from utils.registry import HEADS
    

@HEADS.register(name="DilatedInstanceHead")
class DilatedInstanceHead(nn.Module):
    def __init__(self, 
                 in_channels: int = 256, 
                 num_convs: int = 4, 
                 num_classes: int = 80, 
                 kernel_dim: int = 256, 
                 num_masks: int = 100, 
                 num_groups: int = 1,
                 activation: str = "softmax"):
        # super(DilatedInstanceHead, self).__init__(
        #     in_channels=in_channels, 
        #     num_convs=num_convs, 
        #     num_classes=num_classes, 
        #     kernel_dim=kernel_dim, 
        #     num_masks=num_masks, 
        #     num_groups=num_groups,
        #     activation=activation
        # )
        super().__init__()

        self.dim = in_channels
        self.num_convs = num_convs
        self.num_masks = num_masks
        self.kernel_dim = kernel_dim
        self.num_groups = num_groups
        self.num_classes = num_classes + 1
        self.activation = activation
        
        self.iam_conv_local = nn.Conv2d(self.dim, self.num_masks, 3, padding=1)
        self.iam_conv_global = nn.Conv2d(self.dim, self.num_masks, 3, padding=4, dilation=4)
        
        expand_dim = self.dim * self.num_groups
        self.fc = nn.Linear(expand_dim, expand_dim)

        # outputs
        self.cls_score = nn.Linear(expand_dim, self.num_classes)
        self.mask_kernel = nn.Linear(expand_dim, self.kernel_dim)
        self.objectness = nn.Linear(expand_dim, 1)
        
        self.prior_prob = 0.01
        self._init_weights()
        
        self.softmax_bias = nn.Parameter(torch.ones([1, ]))
        # self.temperature = nn.Parameter(torch.tensor([30.]))
        # self.temperature = 30.0


    def _init_weights(self):
        bias_value = -np.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.iam_conv_local, self.iam_conv_global, self.cls_score]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.iam_conv_local.weight, std=0.01)
        init.normal_(self.iam_conv_global.weight, std=0.01)
        init.normal_(self.cls_score.weight, std=0.01)
        init.normal_(self.mask_kernel.weight, std=0.01)
        init.constant_(self.mask_kernel.bias, 0.0)

        c2_xavier_fill(self.fc)


    def iam_conv(self, features):
        iam_local = self.iam_conv_local(features)
        iam_global = self.iam_conv_global(features)
        iam = iam_local + iam_global
        return iam
    

    def forward(self, features, idx=None):
        print(self.iam_conv)
        # predict instance activation maps
        iam = self.iam_conv(features)

        B, N, H, W = iam.shape
        C = features.size(1)
        
        # BxNxHxW -> BxNx(HW)
        if self.activation == "softmax":
            iam_prob = F.softmax(iam.view(B, N, -1) + self.softmax_bias, dim=-1)
        else:
            raise NotImplementedError(f"No activation {self.activation} found!")
        
        # iam_prob = F.softmax((iam.view(B, N, -1) + self.softmax_bias) / self.temperature, dim=-1)
        # if self.temperature > 1:
        #     self.temperature -= 0.5
        # iam_prob = F.softmax(iam.view(B, N, -1), dim=-1)

        # iam_prob = iam.sigmoid()
        # iam_prob = iam_prob.view(B, N, -1)
        # normalizer = iam_prob.sum(-1).clamp(min=1e-6)
        # iam_prob = iam_prob / normalizer[:, :, None]


        # aggregate features: BxCxHxW -> Bx(HW)xC
        inst_features = torch.bmm(
            iam_prob, features.view(B, C, -1).permute(0, 2, 1))

        inst_features = inst_features.reshape(
            B, self.num_groups, N // self.num_groups, -1).transpose(1, 2).reshape(B, N // self.num_groups, -1)
        
        inst_features = F.relu_(self.fc(inst_features))

        # predict classification & segmentation kernel & objectness
        pred_logits = self.cls_score(inst_features)
        pred_kernel = self.mask_kernel(inst_features)
        pred_scores = self.objectness(inst_features)

        return pred_logits, pred_kernel, pred_scores, iam



if __name__ == "__main__":
    instance_head = DilatedInstanceHead(in_channels=256, num_classes=1, kernel_dim=10, num_masks=5)
    x = torch.rand(1, 256, 512, 512)
    pred_logits, pred_kernel, pred_scores, iam = instance_head(x)
    print(pred_kernel.shape)
    print(pred_logits.shape)
