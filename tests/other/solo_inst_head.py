import torch
import torch.nn.functional as F
from torch import nn

import math



class SOLOv2MaskHead(nn.Module):
    def __init__(self, cfg, input_shape):
        """
        SOLOv2 Mask Head.
        """
        super().__init__()
        # fmt: off
        self.mask_on = True
        self.num_masks = 10
        self.mask_in_features = ["p2", "p3", "p4", "p5", "p6"]
        self.mask_in_channels = 128
        self.mask_channels = 128
        self.num_levels = len(self.mask_in_features)
        # fmt: on
        norm = "GN"

        self.convs_all_levels = nn.ModuleList()
        for i in range(self.num_levels):
            convs_per_level = nn.Sequential()
            if i == 0:
                conv_tower = list()
                conv_tower.append(nn.Conv2d(
                    self.mask_in_channels, self.mask_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None
                ))
                if norm == "GN":
                    conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                conv_tower.append(nn.ReLU(inplace=False))
                convs_per_level.add_module('conv' + str(i), nn.Sequential(*conv_tower))
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    chn = self.mask_in_channels + 2 if i == 3 else self.mask_in_channels
                    conv_tower = list()
                    conv_tower.append(nn.Conv2d(
                        chn, self.mask_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=norm is None
                    ))
                    if norm == "GN":
                        conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                    conv_tower.append(nn.ReLU(inplace=False))
                    convs_per_level.add_module('conv' + str(j), nn.Sequential(*conv_tower))
                    upsample_tower = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module(
                        'upsample' + str(j), upsample_tower)
                    continue
                conv_tower = list()
                conv_tower.append(nn.Conv2d(
                    self.mask_channels, self.mask_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None
                ))
                if norm == "GN":
                    conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                conv_tower.append(nn.ReLU(inplace=False))
                convs_per_level.add_module('conv' + str(j), nn.Sequential(*conv_tower))
                upsample_tower = nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=False)
                convs_per_level.add_module('upsample' + str(j), upsample_tower)

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                self.mask_channels, self.num_masks,
                kernel_size=1, stride=1,
                padding=0, bias=norm is None),
            # nn.GroupNorm(32, self.num_masks),
            # nn.ReLU(inplace=True)
        )



    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            pass
        """

        # bottom features first.
        feature_add_all_level = self.convs_all_levels[0](features[0])
        for i in range(1):
            mask_feat = features[i]
            if i == 3:  # add for coord.
                x_range = torch.linspace(-1, 1, mask_feat.shape[-1], device=mask_feat.device)
                y_range = torch.linspace(-1, 1, mask_feat.shape[-2], device=mask_feat.device)
                y, x = torch.meshgrid(y_range, x_range)
                y = y.expand([mask_feat.shape[0], 1, -1, -1])
                x = x.expand([mask_feat.shape[0], 1, -1, -1])
                coord_feat = torch.cat([x, y], 1)
                mask_feat = torch.cat([mask_feat, coord_feat], 1)
            # add for top features.
            feature_add_all_level = feature_add_all_level + self.convs_all_levels[i](mask_feat)

        mask_pred = self.conv_pred(feature_add_all_level)
        return mask_pred



class SOLOv2InsHead(nn.Module):
    def __init__(self, cfg, input_shape):
        """
        SOLOv2 Instance Head.
        """
        super().__init__()
        # fmt: off
        self.num_classes = 1
        self.num_kernels = 10
        self.num_grids = [7]
        self.instance_in_features = ["p2", "p3", "p4", "p5", "p6"]
        self.instance_strides = [8, 8, 16, 32, 32]
        self.instance_in_channels = 128
        self.instance_channels = 64

        # Convolutions to use in the towers
        self.num_levels = len(self.instance_in_features)

        assert self.num_levels == len(self.instance_strides), \
            print("Strides should match the features.")
        # fmt: on

        head_configs = {"cate": (4,
                                 False,
                                 False),
                        "kernel": (4,
                                   False,
                                   False)
                        }

        norm = "GN"
        # in_channels = [s.channels for s in input_shape]
        # assert len(set(in_channels)) == 1, \
        #     print("Each level must have the same channel!")
        # in_channels = in_channels[0]
        # assert in_channels == self.instance_in_channels, \
        #     print("In channels should equal to tower in channels!")

        for head in head_configs:
            tower = []
            num_convs, use_deformable, use_coord = head_configs[head]
            for i in range(num_convs):
                conv_func = nn.Conv2d
                if i == 0:
                    if use_coord:
                        chn = self.instance_in_channels + 2
                    else:
                        chn = self.instance_in_channels
                else:
                    chn = self.instance_channels

                tower.append(conv_func(
                        chn, self.instance_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=norm is None
                ))
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, self.instance_channels))
                tower.append(nn.ReLU(inplace=True))
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.cate_pred = nn.Conv2d(
            self.instance_channels, self.num_classes,
            kernel_size=3, stride=1, padding=1
        )
        self.kernel_pred = nn.Conv2d(
            self.instance_channels, self.num_kernels,
            kernel_size=3, stride=1, padding=1
        )

        for modules in [
            self.cate_tower, self.kernel_tower,
            self.cate_pred, self.kernel_pred,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cate_pred.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            pass
        """
        cate_pred = []
        kernel_pred = []

        for idx, feature in enumerate(features):
            # print(idx)
            ins_kernel_feat = feature
            # concat coord
            # x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], device=ins_kernel_feat.device)
            # y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], device=ins_kernel_feat.device)
            # y, x = torch.meshgrid(y_range, x_range)
            # y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
            # x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
            # coord_feat = torch.cat([x, y], 1)
            # ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)

            # individual feature.
            kernel_feat = ins_kernel_feat
            seg_num_grid = self.num_grids[idx]
            kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear')
            # cate_feat = kernel_feat[:, :-2, :, :]
            cate_feat = kernel_feat


            # kernel
            kernel_feat = self.kernel_tower(kernel_feat)
            kernel_pred.append(self.kernel_pred(kernel_feat))

            # cate
            cate_feat = self.cate_tower(cate_feat)
            cate_pred.append(self.cate_pred(cate_feat))
        return cate_pred, kernel_pred


if __name__ == "__main__":
    model = SOLOv2MaskHead(None, 32)
    x = [torch.rand(2, 128, 64, 64)]
    mask_pred = model(x)
    print(mask_pred.shape)


    model = SOLOv2InsHead(None, 32)
    print(model)
    x = [torch.rand(2, 128, 64, 64)]
    cate_pred, kernel_preds = model(x)
    for i in kernel_preds:
        print(i.shape)

    for i in cate_pred:
        print(i.shape)


    ins_pred = mask_pred

    kernel_preds = [kernel_preds[0].view(2, 10, -1)]
    # kernel_preds = kernel_preds.view(2, 10, -1)
    print()

    # generate masks
    ins_pred_list = []
    for b_kernel_pred in kernel_preds:
        b_mask_pred = []
        print(f"single kernel batch: {b_kernel_pred.shape}")
        for idx, kernel_pred in enumerate(b_kernel_pred):

            if kernel_pred.size()[-1] == 0:
                continue

            cur_ins_pred = ins_pred[idx, ...]
            H, W = cur_ins_pred.shape[-2:]

            print(f"kernel: {kernel_pred.shape}")
            N, I = kernel_pred.shape
            cur_ins_pred = cur_ins_pred.unsqueeze(0)
            kernel_pred = kernel_pred.permute(1, 0).view(I, -1, 1, 1)
            print(f"1x1 kernel: {kernel_pred.shape}")
            print(f"instance: {cur_ins_pred.shape}")
            cur_ins_pred = F.conv2d(cur_ins_pred, kernel_pred, stride=1).view(-1, H, W)
            print(f"conv instance: {cur_ins_pred.shape}")
            print()

            b_mask_pred.append(cur_ins_pred)
        if len(b_mask_pred) == 0:
            b_mask_pred = None
        else:
            b_mask_pred = torch.cat(b_mask_pred, 0)
        ins_pred_list.append(b_mask_pred)


    

    # pred_cates, pred_kernels = cate_pred, kernel_preds

    # kernel_preds = kernel_preds[0]
    # seg_preds = torch.rand(2, 10, 32, 32)
    # kernel_preds = kernel_preds.view(2, -1, 10)


    # # # kernel: (1, N, D) -> (N, D, 1, 1)
    # masks = []
    # for b in range(len(kernel_preds)):
    #     m = seg_preds[b].unsqueeze(0)
    #     k = kernel_preds[b]

    #     N, D = k.shape
    #     k = k.view(N, D, 1, 1)

    #     inst = F.conv2d(m, k, stride=1)
    #     masks.append(inst)
    # masks = torch.cat(masks, dim=0)

    # print(masks.shape)


    # (2, 64, 32, 32) -> (2, 10, 7, 7) -> (2, 10, 49)
    # (C, H, W) -> (c, h, w) -> (hw, c)
    # (hw, c)

    # (2, 64, 32, 32) -> (2, 10, 32, 32)
    # (2, 10, 32*32) -> ()


    # class DepthwiseConvBlock(nn.Module):
    #     def __init__(self, in_channels, out_channels, kernel_size, num_kernels):
    #         super(DepthwiseConvBlock, self).__init__()
    #         self.depthwise_conv = nn.Conv2d(
    #             in_channels,
    #             out_channels * num_kernels,
    #             kernel_size=kernel_size,
    #             groups=in_channels
    #         )
    #         self.out_channels = out_channels
    #         self.num_kernels = num_kernels

    #     def forward(self, x):
    #         x = self.depthwise_conv(x)
    #         x = x.view(x.size(0), self.num_kernels, self.out_channels)
    #         return x

    # # Usage:
    # input_channels = 64
    # output_channels = 128
    # num_kernels = 10
    # kernel_size = 3

    # conv_block = DepthwiseConvBlock(input_channels, output_channels, kernel_size, num_kernels)
    # output = conv_block(torch.randn(2, input_channels, 32, 32))
    # print(output.shape)  # Should be (2, 10, 128)