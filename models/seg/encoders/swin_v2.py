import os
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformerBlock, PatchEmbed, PatchMerging, default_cfgs
from timm.models.helpers import load_pretrained, build_model_with_cfg
from timm.models.vision_transformer import checkpoint_filter_fn

import sys
sys.path.append("./")

from utils.registry import MODELS

# ---------------------------------------------
# https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L15
class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


@MODELS.register(name="SwinTransformer")
class SwinTransformer(nn.Module):
    def __init__(self, 
                 pretrain_img_size=512,
                 embed_dim=96,
                 patch_size=4,
                 window_size=8,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 pretrained=True,
                 **kwargs):
        super(SwinTransformer, self).__init__()

        self.embed_dim = embed_dim
        self.embed_dims = [embed_dim * 2**i for i in range(len(depths))]
        self.depths = depths
        self.num_heads = num_heads
        self.num_layers = len(depths)
        self.window_size = window_size
        self.patch_size = patch_size
        self.pretrain_img_size = pretrain_img_size
        self.out_indices = out_indices

        self.patch_embed = PatchEmbed(
            img_size=pretrain_img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = nn.ModuleDict({
                'blocks': nn.ModuleList([
                    SwinTransformerBlock(
                        dim=embed_dim * 2**i,
                        input_resolution=(pretrain_img_size // (patch_size * 2**i), pretrain_img_size // (patch_size * 2**i)),
                        num_heads=num_heads[i],
                        window_size=window_size,
                        shift_size=0 if (j % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i]) + j],
                        norm_layer=norm_layer,
                    ) for j in range(depths[i])
                ]),
                'downsample': PatchMerging(
                    input_resolution=(pretrain_img_size // (patch_size * 2**i), pretrain_img_size // (patch_size * 2**i)),
                    dim=embed_dim * 2**i,
                    norm_layer=norm_layer
                ) if i < self.num_layers - 1 else None,
                'norm': norm_layer(embed_dim * 2**i)
            })
            self.layers.append(layer)

        if pretrained:
            self.load_pretrained_weights()

    def load_pretrained_weights(self):
        model_cfg_name = self.get_model_cfg_name()
        if model_cfg_name:
            cfg = default_cfgs[model_cfg_name]
            load_pretrained(self, cfg, strict=False)

        # model_cfg_name = self.get_model_cfg_name()
        # if model_cfg_name:
        #     model_kwargs = {
        #         'pretrain_img_size': self.pretrain_img_size,
        #         'patch_size': self.patch_size,
        #         'window_size': self.window_size, 
        #         'embed_dim': self.embed_dim,
        #         'depths': self.depths,
        #         'num_heads': self.num_heads,
        #         'pretrained': True
        #     }
        #     model = build_model_with_cfg(
        #         SwinTransformerV2, model_cfg_name,
        #         pretrained_filter_fn=checkpoint_filter_fn,
        #         **model_kwargs)
        #     self.load_state_dict(model.state_dict(), strict=True)

    def get_model_cfg_name(self):
        model_cfgs = {
            (96, (2, 2, 6, 2)): 'swin_tiny_patch4_window7_224',
            (96, (2, 2, 18, 2)): 'swin_small_patch4_window7_224',
            (128, (2, 2, 18, 2)): 'swin_base_patch4_window7_224',
            (192, (2, 2, 18, 2)): 'swin_large_patch4_window7_224',
        }
        return model_cfgs.get((self.embed_dim, tuple(self.depths)), None)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        outputs = []
        for i, layer in enumerate(self.layers):
            for blk in layer['blocks']:
                x = blk(x)
            x = layer['norm'](x)
            if i in self.out_indices:
                H, W = self.patch_embed.patches_resolution[0] // (2 ** i), self.patch_embed.patches_resolution[1] // (2 ** i)
                x_out = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                outputs.append(x_out)
            if layer['downsample'] is not None:
                x = layer['downsample'](x)
        
        return outputs


if __name__ == '__main__':
    model = SwinTransformer(pretrain_img_size=512, window_size=8, pretrained=True)
    x = torch.rand(1, 3, 512, 512)
    feats = model(x)
    for y in feats:
        print(y.shape)
