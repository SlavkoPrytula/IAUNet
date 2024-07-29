import torch 
from torch import nn
from torch.nn import functional as F
import numpy as np

import sys
sys.path.append("./")

# inst = torch.rand(1, 30, 10, 10)
# kernel = torch.rand(1, 1, 10, 30)

# # filters = torch.randn(1, 1, 3, 3)
# # kernel_pred = kernel.permute(1, 0).view(1, -1, 1, 1)
# # inputs = torch.randn(1, 1, 5, 5)
# # print(F.conv2d(inputs, filters, padding=1).shape)


# # H, W = inst.shape[-2:]
# # N, I = kernel.shape

# print(inst.shape, kernel.shape)

# # inst = inst.unsqueeze(0)
# # kernel = kernel.permute(1, 0).view(I, -1, 1, 1)
# inst = F.conv2d(inst, kernel, stride=1) #.view(-1, H, W)

# print(inst.shape)



# mask = torch.rand(2, 30, 10, 10)
# kernel = torch.rand(2, 5, 30)

# b_mask = []
# for b in range(len(kernel)):
#     m = mask[b].unsqueeze(0)
#     k = kernel[b]

#     N, D = k.shape
#     k = k.view(N, D, 1, 1)

#     inst = F.conv2d(m, k, stride=1)
#     b_mask.append(inst)

# b_mask = torch.cat(b_mask, dim=0)
# print(b_mask.shape)

# feats = torch.rand(1, 50, 10, 10)

# feats = nn.Conv2d(50, 5, 3, 1, 1)(feats)
# print(feats.shape)
# cls = nn.Linear(100, 1)(feats.view(1, 5, -1))
# print(cls.shape)
# feats = nn.AdaptiveAvgPool2d((3, 3))(feats)
# print(feats.shape)


# feats1 = torch.rand(1, 100, 10, 10)
# feats2 = torch.rand(1, 50, 10, 10)
# B, N, H, W = feats2.shape
# C = feats1.shape[1]


# print(
#     torch.matmul(
#         feats2.view(B, N, -1), 
#         feats1.view(B, C, -1).permute(0, 2, 1)
#         ).shape
#     )


# conv = nn.Conv2d(30, 5, 3, 1, 1)

# mask = torch.zeros(1, 30, 10, 10)
# kernel = torch.zeros(1, 5*9, 30)
# # kernel = kernel.view(5, 30, 1, 1)
# kernel = kernel.view(5, 30, 3, 3)
# # kernel = torch.randn(5, 30, 3, 3) 

# # kernel = torch.nn.Parameter(kernel, requires_grad=False)

# # inst = F.conv2d(mask, kernel, stride=1, padding=1)
# with torch.no_grad():
#     conv.weight.copy_(kernel)
# # conv.weight = kernel
# # print(conv.bias.shape)
# inst = conv(mask)

# # inst = mask * kernel 
# print(inst.shape) # torch.Size([1, 5, 10, 10])


from utils.tiles import fold, unfold




mask = torch.zeros(2, 30, 20, 20)
B, C, H, W = mask.shape
num_grids = 4

N = H*W // ((H//num_grids)*(W//num_grids))
print(N)

kernel_pool = nn.ModuleList([])
for _ in range(N):
    kernel_pool.append(
        nn.Sequential(
            nn.AdaptiveAvgPool2d(3),
        )
    )


iam = []
for b_mask in mask:
    b_mask = b_mask.unsqueeze(0)
    partitioned_mask = unfold(b_mask, H//num_grids, W//num_grids)
    partitioned_kernels = []

    for i, p_m in enumerate(partitioned_mask):
        k = kernel_pool[i](p_m)
        partitioned_kernels.append(k)
    kernel = torch.cat(partitioned_kernels, 0)

    _iam = F.conv2d(b_mask, kernel, stride=1, padding=1)

    iam.append(_iam)

iam = torch.cat(iam)
print(iam.shape)




# import torch.nn as nn

# class Attention(nn.Module):
#     def __init__(self, in_dim, out_channels=5):
#         super(Attention, self).__init__()
#         self.chanel_in = in_dim
#         self.out_channels = out_channels
        
#         self.key_conv = nn.Conv2d(in_channels=self.chanel_in, out_channels=5, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=self.chanel_in, out_channels=self.out_channels, kernel_size=1)
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  

#         self.softmax = nn.Softmax(dim=-1)
    
#     def forward(self, x):
#         x = self.maxpool(x)
#         x = self.maxpool(x)
#         B, C, W, H = x.size()

#         key = self.key_conv(x).view(-1, 1, H*W)      # (BN, 1, HW)
#         value = self.value_conv(x).view(-1, H*W, 1)  # (BN, HW, 1)

#         energy = torch.matmul(value, key)
#         attn = self.softmax(energy) # (HW, HW)

#         attn = attn.mean(1).view(B, -1, H, W)
#         attn = F.interpolate(attn, size=[H*4, W*4], mode='bilinear', align_corners=False)

#         return attn



# class CentextAttention(nn.Module):
#     def __init__(self, in_channels, out_channels=5):
#         super(CentextAttention, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         self.conv_mask = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
    
#     def forward(self, x):
#         context_mask = self.conv_mask(x)
        
#         return context_mask





# class ContextBlock(nn.Module):

#     def __init__(self,
#                  inplanes,
#                  ratio,
#                  pooling_type='att',
#                  fusion_types=('channel_add', )):
#         super(ContextBlock, self).__init__()
#         assert pooling_type in ['avg', 'att']
#         assert isinstance(fusion_types, (list, tuple))
#         valid_fusion_types = ['channel_add', 'channel_mul']
#         assert all([f in valid_fusion_types for f in fusion_types])
#         assert len(fusion_types) > 0, 'at least one fusion should be used'
#         self.inplanes = inplanes
#         self.ratio = ratio
#         self.planes = int(inplanes * ratio)
#         self.pooling_type = pooling_type
#         self.fusion_types = fusion_types
#         if pooling_type == 'att':
#             self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
#             self.softmax = nn.Softmax(dim=2)
#         else:
#             self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         if 'channel_add' in fusion_types:
#             self.channel_add_conv = nn.Sequential(
#                 nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
#                 nn.LayerNorm([self.planes, 1, 1]),
#                 nn.ReLU(inplace=True),  # yapf: disable
#                 nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
#         else:
#             self.channel_add_conv = None
#         if 'channel_mul' in fusion_types:
#             self.channel_mul_conv = nn.Sequential(
#                 nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
#                 nn.LayerNorm([self.planes, 1, 1]),
#                 nn.ReLU(inplace=True),  # yapf: disable
#                 nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
#         else:
#             self.channel_mul_conv = None
#         self.reset_parameters()

#     def reset_parameters(self):
#         if self.pooling_type == 'att':
#             kaiming_init(self.conv_mask, mode='fan_in')
#             self.conv_mask.inited = True

#         if self.channel_add_conv is not None:
#             last_zero_init(self.channel_add_conv)
#         if self.channel_mul_conv is not None:
#             last_zero_init(self.channel_mul_conv)

#     def spatial_pool(self, x):
#         batch, channel, height, width = x.size()
#         if self.pooling_type == 'att':
#             input_x = x
#             # [N, C, H * W]
#             input_x = input_x.view(batch, channel, height * width)
#             # [N, 1, C, H * W]
#             input_x = input_x.unsqueeze(1)
#             # [N, 1, H, W]
#             context_mask = self.conv_mask(x)
#             # [N, 1, H * W]
#             context_mask = context_mask.view(batch, 1, height * width)
#             # [N, 1, H * W]
#             context_mask = self.softmax(context_mask)
#             # [N, 1, H * W, 1]
#             context_mask = context_mask.unsqueeze(-1)
#             # [N, 1, C, 1]
#             context = torch.matmul(input_x, context_mask)
#             # [N, C, 1, 1]
#             context = context.view(batch, channel, 1, 1)
#         else:
#             # [N, C, 1, 1]
#             context = self.avg_pool(x)

#         return context

#     def forward(self, x):
#         # [N, C, 1, 1]
#         context = self.spatial_pool(x)

#         out = x
#         if self.channel_mul_conv is not None:
#             # [N, C, 1, 1]
#             channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
#             out = out * channel_mul_term
#         if self.channel_add_conv is not None:
#             # [N, C, 1, 1]
#             channel_add_term = self.channel_add_conv(context)
#             out = out + channel_add_term

#         return out


# mask1 = torch.rand(1, 10, 512, 512)
# # mask2 = torch.rand(1, 1, 256, 256)


# # batch_size, num_channels, height, width = mask1.shape
# # mask1_reshaped = mask1.view(batch_size, num_channels, -1)  # Reshape to (batch_size, num_channels, H*W)

# # scores1 = torch.matmul(mask1_reshaped, mask1_reshaped.permute(0, 2, 1))  # Self-attention for mask1
# # attention_map1 = F.softmax(scores1, dim=-1)


# # out = torch.matmul(
# #     mask1.view, mask2.view()
# #     )

# attn = CentextAttention(10, 5)(mask1)

# print(attn.shape)