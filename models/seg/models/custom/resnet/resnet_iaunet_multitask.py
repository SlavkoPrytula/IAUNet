# copied from 45966022
import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn.weight_init import c2_msra_fill

import sys
sys.path.append("./")

# from models.seg.heads.instance_head import InstanceHead, InstanceBranch
# from models.seg.heads.mask_head import MaskBranch

# from models.seg.models.base import DoubleConv, SE_block
# from models.seg.blocks.tests import (PyramidPooling, PyramidPooling_v3, PyramidPooling_v5, 
#                              DoubleConv_v2, DoubleConv_v3, DoubleConvModule)


from ....heads.instance_head import InstanceHead, InstanceBranch
from ....heads.mask_head import MaskBranch

from models.seg.models.base import BaseModel
from models.seg.models.base import DoubleConv, SE_block
from ....nn.blocks import (PyramidPooling, PyramidPooling_v3, PyramidPooling_v5, 
                             DoubleConv_v2, DoubleConv_v3, DoubleConvModule)

from configs import cfg
from utils.registry import MODELS, HEADS

import torchvision


@MODELS.register(name="resnet_iaunet_multitask")
class IAUNet(nn.Module):
    def __init__(
        self,
        cfg: cfg,
        embed_dims=[64, 256, 512, 1024, 2048],
        # embed_dims=[64, 64, 128, 256, 512],
        # embed_dims=[64, 128, 256, 512, 1024],
        # embed_dims=[32, 96, 192, 384, 768],
        # embed_dims=[32, 64, 128, 256, 512],
        # embed_dims=[64, 64, 64, 64, 64],
        pyramid_pooling=True,
        n_pp_features=128,
    ):
        super().__init__()
        
        self.coord_conv = cfg.model.coord_conv
        self.multi_level = cfg.model.multi_level
        self.kernel_dim = cfg.model.instance_head.kernel_dim
        self.num_convs = cfg.model.num_convs

        self.cfg = cfg  
        self.n_input_channels = cfg.model.in_channels
        self.n_output_channels = cfg.model.out_channels
        self.n_levels = cfg.model.n_levels

        self.embed_dims = embed_dims
        self.n_pp_features = n_pp_features
        self.pyramid_pooling = pyramid_pooling
        self.kernel_strides_map = {1: 16, 2: 8, 3: 4, 4: 2, 5: 1}

        self.skips = True

        self.down_conv_layers = nn.ModuleList([])
        self.down_pp_layers = nn.ModuleList([])
        self.up_conv_layers = nn.ModuleList([])
        self.up_conv_layers = nn.ModuleList([])


        encoder = torchvision.models.resnet50(pretrained=True)

        self.firstlayer = nn.Sequential(*list(encoder.children())[:3])
        self.maxpool = list(encoder.children())[3]
        self.encoder1 = encoder.layer1
        self.encoder2 = encoder.layer2
        self.encoder3 = encoder.layer3
        self.encoder4 = encoder.layer4

        self.bridge = nn.Sequential(
            # nn.Conv2d(embed_dims[4], embed_dims[4], kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(embed_dims[4]),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv_v2(embed_dims[4], embed_dims[4]),
            SE_block(num_features=embed_dims[4]),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

        
        embed_dims = self.embed_dims[::-1]
        for i in range(self.n_levels):
            in_channels = embed_dims[i] * 2 + 2

            if i != self.n_levels-1:
                out_channels = embed_dims[i+1]
            else:
                out_channels = embed_dims[i]

            upconv = nn.Sequential(
                DoubleConv_v2(in_channels, out_channels),
                SE_block(num_features=out_channels)
            )
            self.up_conv_layers.append(upconv)


        embed_dims = embed_dims[1:] + [embed_dims[-1]]

        # mask branch.
        mask_dim = cfg.model.mask_dim
        self.mask_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                self.mask_branch.append(MaskBranch(embed_dims[i], out_channels=mask_dim, num_convs=self.num_convs))
            else:
                self.mask_branch.append(MaskBranch(embed_dims[i] + mask_dim, out_channels=mask_dim, num_convs=self.num_convs))

        self.projection = nn.Conv2d(mask_dim, self.kernel_dim, kernel_size=1)
        c2_msra_fill(self.projection)
        
        # instance features.
        self.prior_instance_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                self.prior_instance_branch.append(InstanceBranch(in_channels=embed_dims[i] + 2, out_channels=mask_dim, num_convs=self.num_convs))
            else:
                self.prior_instance_branch.append(InstanceBranch(in_channels=embed_dims[i] + mask_dim + 2, out_channels=mask_dim, num_convs=self.num_convs))

        # instance branch.
        self.instance_head = HEADS.build(cfg.model.instance_head)


        for modules in [self.down_conv_layers, self.up_conv_layers, self.down_pp_layers]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)


    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        x_loc = torch.linspace(-1, 1, h, device=x.device)
        y_loc = torch.linspace(-1, 1, w, device=x.device)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x_loc, y_loc], 1)

        return coord_feat
        

    # TESTING: add instance and mask branches only to the final layer of the decoder
    def forward(self, x, idx=None):
        
        # go down
        e1 = self.firstlayer(x)
        maxe1 = self.maxpool(e1)
        e2 = self.encoder1(maxe1)
        e3 = self.encoder2(e2)
        e4 = self.encoder3(e3)
        e5 = self.encoder4(e4)

        skips = [e1, e2, e3, e4, e5]

        # middle
        x = self.bridge(e5)

        # go up
        for i in range(self.n_levels):
            if i != 0:
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
            coord_features = self.compute_coordinates(x)
            x = torch.cat([coord_features, x], dim=1)

            x = torch.cat([x, skips[-(i + 1)]], dim=1)
            x = self.up_conv_layers[i](x)

            
            # multi-level
            # if self.multi_level:
            if i != 0:
                mask_feats = nn.UpsamplingBilinear2d(scale_factor=2)(mask_feats)    # (1, 128, 128, 128)
                mask_feats = torch.cat([x, mask_feats], dim=1)
                mask_feats = self.mask_branch[i](mask_feats)
            else:
                mask_feats = self.mask_branch[i](x)

            if i != 0:
                # scale: (B, N, Hx, Wx) -> (B, N, Hx * 2, Wx * 2)
                # x features shape: (B, Di, Hx * 2, Wx * 2)
                inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
                inst_feats = torch.cat([x, inst_feats], dim=1)
                coord_features = self.compute_coordinates(inst_feats)

                inst_feats = torch.cat([coord_features, inst_feats], dim=1)
                inst_feats = self.prior_instance_branch[i](inst_feats)
            else:
                # inst_feats shape: (B, Dm, Hx, Wx)
                coord_features = self.compute_coordinates(x)
                inst_feats = torch.cat([coord_features, x], dim=1)
                inst_feats = self.prior_instance_branch[i](inst_feats)
                    

        # out layer.
        results = self.instance_head(inst_feats)

        logits = results["logits"]
        mask_kernel = results["mask_kernel"]
        # border_kernel = results["border_kernel"]
        scores = results["objectness_scores"]
        bboxes = results["bboxes"]
        iam = results["iam"]

        mask_feats = self.projection(mask_feats)

        
        # Predicting instance masks
        N = mask_kernel.shape[1]
        B, C, H, W = mask_feats.shape

        inst_masks = torch.bmm(mask_kernel, mask_feats.view(B, C, H * W))
        inst_masks = inst_masks.view(B, N, H, W)

        # borders_masks = torch.bmm(border_kernel, mask_feats.view(B, C, H * W))
        # borders_masks = borders_masks.view(B, N, H, W)
        
        bboxes = bboxes.sigmoid()

        inst_masks = nn.UpsamplingBilinear2d(scale_factor=2)(inst_masks)
        # borders_masks = nn.UpsamplingBilinear2d(scale_factor=2)(borders_masks)
        iam = nn.UpsamplingBilinear2d(scale_factor=2)(iam)


        output = {
            'pred_logits': logits,
            'pred_scores': scores,
            'pred_iam': iam,
            'pred_masks': inst_masks,
            # 'pred_borders_masks': borders_masks,
            'pred_bboxes': bboxes,
        }
    
        return output
    

if __name__ == "__main__":
    import time 

    model = IAUNet(cfg)
    x = torch.rand(1, 3, 512, 512)
    
    time_s = time.time()
    out = model(x)
    print(out["pred_masks"].shape)
    time_e = time.time()
    print(f'loaded in {time_e - time_s}(s)')