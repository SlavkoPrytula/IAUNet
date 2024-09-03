# copied from 45966022
import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn.weight_init import c2_msra_fill

import sys
sys.path.append("./")

from ....heads.instance_head import InstanceBranch
from ....heads.mask_head import MaskBranch
from ....nn.blocks import DoubleConv_v2, SE_block

from configs.structure import cfg
from utils.registry import MODELS, HEADS

import torchvision


@MODELS.register(name="resnet_iaunet_multitask_ml")
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
        
        self.coord_conv = cfg.model.decoder.coord_conv
        self.num_convs = cfg.model.decoder.num_convs

        self.mask_dim = cfg.model.decoder.mask_branch.dim
        self.inst_dim = cfg.model.decoder.instance_branch.dim
        self.kernel_dim = cfg.model.decoder.instance_head.kernel_dim

        self.cfg = cfg  
        self.n_levels = cfg.model.n_levels

        self.embed_dims = embed_dims
        self.skips = True

        encoder = torchvision.models.resnet50(pretrained=True)

        self.firstlayer = nn.Sequential(*list(encoder.children())[:3])
        self.maxpool = list(encoder.children())[3]
        self.encoder1 = encoder.layer1
        self.encoder2 = encoder.layer2
        self.encoder3 = encoder.layer3
        self.encoder4 = encoder.layer4

        self.bridge = nn.Sequential(
            nn.Conv2d(embed_dims[4], embed_dims[4], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dims[4]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            
        )
        
        embed_dims = self.embed_dims[::-1]
        
        self.up_conv_layers = nn.ModuleList([])
        for i in range(self.n_levels):
            in_channels = embed_dims[i] * 2 + 2
            out_channels = embed_dims[i+1]

            upconv = nn.Sequential(
                DoubleConv_v2(in_channels, out_channels),
                SE_block(num_features=out_channels)
            )
            self.up_conv_layers.append(upconv)


        embed_dims = embed_dims[1:] + [embed_dims[-1]]

        # mask branch.
        mask_dim = cfg.model.decoder.mask_branch.dim
        mask_branch_layer = HEADS.get(cfg.model.decoder.mask_branch.type)
        self.mask_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                mask_branch = mask_branch_layer(
                    in_channels=embed_dims[i], 
                    out_channels=self.mask_dim, 
                    num_convs=self.num_convs
                )
            else:
                mask_branch = mask_branch_layer(
                    in_channels=embed_dims[i] + self.mask_dim, 
                    out_channels=self.mask_dim, 
                    num_convs=self.num_convs
                )
            self.mask_branch.append(mask_branch)

        self.projection = nn.Conv2d(mask_dim, self.kernel_dim, kernel_size=1)
        c2_msra_fill(self.projection)
        
        # instance features.
        self.instance_branch = nn.ModuleList([])
        instance_branch_layer = HEADS.get(cfg.model.decoder.instance_branch.type)
        for i in range(self.n_levels):
            if i == 0:
                instance_branch = instance_branch_layer(
                    in_channels=embed_dims[i], 
                    out_channels=self.inst_dim, 
                    num_convs=self.num_convs
                )
            else:
                instance_branch = instance_branch_layer(
                    in_channels=embed_dims[i] + self.inst_dim + 2, 
                    out_channels=self.inst_dim, 
                    num_convs=self.num_convs
                )
            self.instance_branch.append(instance_branch)

        # instance branch.
        self.instance_head = nn.ModuleList([])
        for i in range(self.n_levels):
            self.instance_head.append(HEADS.build(cfg.model.decoder.instance_head))

        for modules in [self.up_conv_layers, self.bridge]:
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
        

    def forward(self, x, idx=None):
        ori_shape = x.shape
        
        # go down
        e1 = self.firstlayer(x)
        maxe1 = self.maxpool(e1)
        e2 = self.encoder1(maxe1)
        e3 = self.encoder2(e2)
        e4 = self.encoder3(e3)
        e5 = self.encoder4(e4)

        skips = [e1, e2, e3, e4, e5]

        x = self.bridge(e5)

        # go up
        for i in range(self.n_levels):
            coord_features = self.compute_coordinates(x)
            x = torch.cat([coord_features, x], dim=1)
            
            x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
            x = torch.cat([x, skips[-(i + 1)]], dim=1)
            x = self.up_conv_layers[i](x)

            
            if i != 0:
                mask_feats = nn.UpsamplingBilinear2d(scale_factor=2)(mask_feats)
                mask_feats = torch.cat([x, mask_feats], dim=1)
                mask_feats = self.mask_branch[i](mask_feats)   
            else:
                mask_feats = self.mask_branch[i](x)


            if i != 0:
                inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
                inst_feats = torch.cat([x, inst_feats], dim=1)

                coord_features = self.compute_coordinates(inst_feats)
                inst_feats = torch.cat([coord_features, inst_feats], dim=1)
                inst_feats = self.instance_branch[i](inst_feats)
            else:
                inst_feats = self.instance_branch[i](x)


            if i != 0:
                results = self.instance_head[i](inst_feats, mask_feats, inst_embed)
                # inst_embed = results["kernels"]['instance_kernel']
                inst_embed = results["inst_feats"]['instance_feats']
            else:
                results = self.instance_head[i](inst_feats, mask_feats)
                # inst_embed = results["kernels"]['instance_kernel']
                inst_embed = results["inst_feats"]['instance_feats']

            mask_feats = results['mask_pixel_feats']
            inst_feats = results['inst_pixel_feats']

        
        mask_feats = self.projection(mask_feats)
        results["mask_feats"] = mask_feats
        results["inst_feats"] = inst_feats

        results = self.process_outputs(results, ori_shape)
    
        return results
    

    def process_outputs(self, results, ori_shape):
        logits = results["logits"]
        scores = results["objectness_scores"]
        inst_kernel = results["kernels"]["instance_kernel"]
        bboxes = results["bboxes"]['instance_bboxes']
        mask_feats = results["mask_feats"]
        inst_feats = results["inst_feats"]
        
        # instance masks.
        N = inst_kernel.shape[1]
        B, C, H, W = mask_feats.shape

        inst_masks = torch.bmm(inst_kernel, mask_feats.view(B, C, H * W))
        inst_masks = inst_masks.view(B, N, H, W)
        bboxes = bboxes.sigmoid()

        inst_masks = F.interpolate(inst_masks, size=ori_shape[-2:], 
                                   mode="bilinear", align_corners=False)

        output = {
            'pred_logits': logits,
            'pred_scores': scores,
            'pred_iams': results['iams'],
            'pred_instance_masks': inst_masks,
            'pred_bboxes': bboxes,
            'pred_instance_feats': {
                "mask_feats": mask_feats,
                "inst_feats": inst_feats
            }
        }
    
        return output