import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InterLevelAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., kv_dim=None,
                 store_attn=False, top_down_update=False, use_euc_dist=False, stride=2, patch_size=(4, 4), dim_reduc=1):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.attn = None
        self.store_attn = store_attn
        self.top_down_update = top_down_update
        self.use_euc_dist = use_euc_dist

        self.patch_size = patch_size
        self.stride = stride

        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.attn_dim = dim // dim_reduc if kv_dim > dim else kv_dim // dim_reduc

        self.q = nn.Linear(dim, self.attn_dim, bias=qkv_bias)
        self.kv = nn.Linear(kv_dim, 2 * self.attn_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relative_position_init = nn.Parameter(torch.zeros(patch_size[0] * patch_size[1]))
        self.relative_position_bias_table = nn.Parameter(torch.zeros(patch_size[0]*patch_size[1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, Xt, Xb, Ht, Wt, Hb, Wb):

        B, _, C = Xb.shape
        patch_size, stride, num_heads = self.patch_size, self.stride, self.num_heads
        Xbp = Xb.permute(0, 2, 1).view(Xb.shape[0], -1, Hb, Wb)
        Xbp = F.unfold(Xbp, kernel_size=patch_size, stride=stride,
                       padding=((patch_size[0] - 1) // 2, (patch_size[1] - 1) // 2)).contiguous()
        Xbp = Xbp.reshape(B, C, patch_size[0] ** 2, Ht * Wt).permute(0, 3, 2, 1)
        Xbp = Xbp.reshape(B * Ht * Wt, -1, C)
        Xt = Xt.view(-1, 1, Xt.shape[-1])

        if self.top_down_update:
            x, y = Xbp, Xt
            H, W = Hb, Wb
        else:
            x, y = Xt, Xbp
            H, W = Ht, Wt

        BHW, N, _ = x.shape
        C = self.attn_dim
        q = self.q(x).reshape(BHW, N, num_heads, C // num_heads).permute(0, 2, 1, 3)
        kv = self.kv(y).reshape(BHW, -1, 2, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        if not self.use_euc_dist:
            m = torch.matmul(q, k.transpose(-2, -1))
        else:
            m = torch.cdist(q, k, compute_mode='use_mm_for_euclid_dist')
        attn = m * self.scale

        if self.top_down_update:
            relative_position_init = self.relative_position_init.unsqueeze(1)
            relative_position_bias = self.relative_position_bias_table.unsqueeze(1)
            attn = attn + relative_position_init + relative_position_bias
            attn = self.patch_wise_softmax(attn, Ht, Wt, Hb, Wb)
        else:
            relative_position_init = self.relative_position_init.unsqueeze(0)
            relative_position_bias = self.relative_position_bias_table.unsqueeze(0)
            attn = attn + relative_position_init + relative_position_bias
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        if self.store_attn:
            self.attn = attn

        x = (attn @ v).transpose(1, 2).reshape(BHW, N, C)

        B = BHW // Ht // Wt
        if self.top_down_update:
            x = x.view(B, Ht * Wt, -1, C)
            x = x.permute(0, 3, 2, 1).reshape(B, C * patch_size[0] ** 2, Ht * Wt)
            x = F.fold(x, output_size=(H, W), kernel_size=patch_size, stride=stride,
                       padding=((patch_size[0] - 1) // 2, (patch_size[1] - 1) // 2)).contiguous()
            x = x.permute(0, 2, 3, 1)
        x = x.view(B, H * W, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def patch_wise_softmax(self, attn, Ht, Wt, Hb, Wb):
        patch_size, stride, num_heads = self.patch_size, self.stride, self.num_heads
        padding = ((patch_size[0] - 1) // 2, (patch_size[1] - 1) // 2)

        attn = torch.exp(attn)
        attn_sum = attn.reshape(-1, Ht * Wt, num_heads, patch_size[0] ** 2)
        attn_sum = attn_sum.permute(0, 2, 3, 1).reshape(-1, patch_size[0] ** 2, Ht * Wt)
        attn_sum = F.fold(attn_sum, output_size=(Hb, Wb), kernel_size=patch_size, stride=stride, padding=padding)
        attn_sum = F.unfold(attn_sum, kernel_size=patch_size, stride=stride, padding=padding).permute(0, 2, 1)

        attn_sum[attn_sum == 0] = 1.0
        return attn / attn_sum.reshape(-1, num_heads, patch_size[0] ** 2, 1)


class TopDownAttn(nn.Module):

    def __init__(self, top_dim, bot_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, store_attn=False,
                 patch_size=4, stride=2, use_euc_dist=False, info_weights=(1, 1), dim_reduc=1):
        super().__init__()

        self.info_weights = info_weights
        self.topdown_attn = InterLevelAttention(
            bot_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, patch_size=to_2tuple(patch_size),
            stride=stride, attn_drop=attn_drop, proj_drop=drop, kv_dim=top_dim, store_attn=store_attn,
            dim_reduc=dim_reduc, top_down_update=True, use_euc_dist=use_euc_dist)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(bot_dim)
        self.norm2 = norm_layer(top_dim)
        self.norm3 = norm_layer(bot_dim)

        mlp_hidden_dim = int(bot_dim * mlp_ratio)
        self.bottom_mlp = Mlp(in_features=bot_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, Xt, Xb, Ht, Wt, Hb, Wb):
        Xb = Xb + self.drop_path(self.topdown_attn(self.norm2(Xt), self.norm1(Xb), Ht, Wt, Hb, Wb))
        Xb_ffn = self.drop_path(self.bottom_mlp(self.norm3(Xb), Hb, Wb))
        Xb = self.info_weights[0]*Xb + self.info_weights[1]*Xb_ffn
        return Xt, Xb


class BottomUpAttn(nn.Module):
    def __init__(self, top_dim, bot_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, store_attn=False,
                 patch_size=4, stride=2, use_euc_dist=False, info_weights=(1, 1), dim_reduc=1):
        super().__init__()

        self.info_weights = info_weights
        self.bottomup_attn = InterLevelAttention(
            top_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, patch_size=to_2tuple(patch_size),
            stride=stride, attn_drop=attn_drop, proj_drop=drop, kv_dim=bot_dim, store_attn=store_attn,
            dim_reduc=dim_reduc, use_euc_dist=use_euc_dist)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm4 = norm_layer(top_dim)
        self.norm5 = norm_layer(bot_dim)
        self.norm6 = norm_layer(top_dim)

        mlp_hidden_dim = int(top_dim * mlp_ratio)
        self.top_mlp = Mlp(in_features=top_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, Xt, Xb, Ht, Wt, Hb, Wb):
        # TODO: check result different for other models
        Xt = Xt + self.drop_path(self.bottomup_attn(self.norm4(Xt), self.norm5(Xb), Ht, Wt, Hb, Wb))
        Xt_ffn = self.drop_path(self.top_mlp(self.norm6(Xt), Ht, Wt))
        Xt = self.info_weights[0] * Xt + self.info_weights[1] * Xt_ffn
        return Xt, Xb


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)

        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 store_attn=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.attn = None
        self.store_attn = store_attn

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if self.store_attn:
            self.attn = attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, store_attn=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, store_attn=store_attn)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                          padding=((patch_size[0] - 1) // 2, (patch_size[1] - 1) // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class HILASegformer(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], use_euc_dist=False, hila_attn=[], td_qk_scale=None,
                 hila_num_heads=[1, 2, 4, 8], share_td_weights=True, share_bu_weights=True, share_bottom_weights=True,
                 info_weights=(1, 1), hila_stride=[1, 1, 1, 1], hila_mlp_ratios=[4, 4, 4, 4], hila_patch_size=4,
                 imagenet_pretraining=False, reuse_bottomsa_pretraining=False, store_attn=False):
        super().__init__()

        self.store_attn = store_attn
        self.num_classes = num_classes
        self.depths = depths
        self.use_euc_dist= use_euc_dist
        self.hila_attn = hila_attn
        self.share_td_weights = share_td_weights
        self.share_bu_weights = share_bu_weights
        self.share_bottom_weights = share_bottom_weights
        self.imagenet_pretraining = imagenet_pretraining
        self.hila_stride = {'12': hila_stride[1], '23': hila_stride[2], '34': hila_stride[3]}
        self.reuse_bottomsa_pretraining = reuse_bottomsa_pretraining
        self.register_buffer('reuse_bottomsa_pretraining_flag', torch.zeros(1))

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1], store_attn=self.store_attn)
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2], store_attn=self.store_attn)
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # HILA block
        self.attn = {}
        self.topdown, self.bottomup = nn.ModuleDict(), nn.ModuleDict()
        self.prevblock, self.prevnorm = nn.ModuleDict(), nn.ModuleDict()
        for i, pair in enumerate(self.hila_attn):
            b, t = pair
            topdown, bottomup, prevblock, prevnorm = self.init_interlayer_attention(
                b-1, t-1, hila_num_heads[b-1], dpr, cur, embed_dims=embed_dims, num_heads=num_heads, mlp_ratios=mlp_ratios,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                norm_layer=nn.LayerNorm, depths=depths, sr_ratios=sr_ratios, use_euc_dist=use_euc_dist,
                td_qk_scale=td_qk_scale, info_weights=info_weights, hila_mlp_ratios=hila_mlp_ratios,
                patch_size=hila_patch_size, hila_stride=hila_stride)
            tag = str(b) + str(t)
            self.topdown.update({tag: topdown})
            self.bottomup.update({tag: bottomup})
            self.prevblock.update({tag: prevblock})
            self.prevnorm.update({tag: prevnorm})

        # classification head
        if self.imagenet_pretraining:
            self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def init_interlayer_attention(self, bottom_idx, top_idx, num_head, dpr, cur, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
                 use_euc_dist=False, patch_size=4, stride=2, td_qk_scale=None, info_weights=(1, 1),
                 hila_stride=[1, 1, 1, 1], hila_mlp_ratios=[4, 4, 4, 4]):

        drop_path = dpr[0]
        td_block_depth = depths[top_idx] // hila_stride[top_idx] - 1 if not self.share_td_weights else 1
        bu_block_depth = depths[top_idx] // hila_stride[top_idx] if not self.share_bu_weights else 1
        sa_block_depth = depths[top_idx] // hila_stride[top_idx] - 1 if not self.share_bottom_weights else 1

        topdown = nn.ModuleList([TopDownAttn(
            top_dim=embed_dims[top_idx], bot_dim=embed_dims[bottom_idx], num_heads=num_head, mlp_ratio=hila_mlp_ratios[top_idx],
            qkv_bias=qkv_bias, qk_scale=td_qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path,
            norm_layer=norm_layer, store_attn=self.store_attn, use_euc_dist=use_euc_dist, patch_size=patch_size,
            stride=stride, info_weights=info_weights).to(device)
            for i in range(td_block_depth)])
        bottomup = nn.ModuleList([BottomUpAttn(
            top_dim=embed_dims[top_idx], bot_dim=embed_dims[bottom_idx], num_heads=num_head, mlp_ratio=hila_mlp_ratios[top_idx],
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path,
            norm_layer=norm_layer, store_attn=self.store_attn, use_euc_dist=use_euc_dist, patch_size=patch_size,
            stride=stride, info_weights=info_weights).to(device)
            for i in range(bu_block_depth)])
        bot_sa = nn.ModuleList([Block(
            dim=embed_dims[bottom_idx], num_heads=num_heads[bottom_idx], mlp_ratio=mlp_ratios[bottom_idx],
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path, norm_layer=norm_layer,
            sr_ratio=sr_ratios[bottom_idx], store_attn=self.store_attn).to(device)
            for i in range(sa_block_depth)])
        bot_norm = norm_layer(embed_dims[bottom_idx])
        return topdown, bottomup, bot_sa, bot_norm

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        return
        # if isinstance(pretrained, str):
        #     load_checkpoint(self, pretrained, map_location='cpu', strict=False)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    def _extract_deform_loss(self, x, mask=None):
        loss = torch.zeros(1, device='cuda')
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def reinitialize_bottomsa_weights(self):
        for i, (block, idx) in enumerate(zip([self.block1, self.block2, self.block3], ['12', '23', '34'])):
            if idx in self.prevblock:
                block_weights = block[-1]
                for j in range(len(self.prevblock[idx])):
                    self.prevblock[idx][j] = deepcopy(block_weights)

    def forward_features(self, x):

        if self.store_attn:
            self.attn = {}
            for stage in ['12', '23', '34', '23_after']:
                self.attn[stage] = {}
                for type in ['top', 'bottom']:
                    self.attn[stage][type] = []
        if self.reuse_bottomsa_pretraining and self.reuse_bottomsa_pretraining_flag == 0:
            self.reuse_bottomsa_pretraining = False
            self.reuse_bottomsa_pretraining_flag = torch.ones(1)
            self.reinitialize_bottomsa_weights()

        B = x.shape[0]
        outs = []

        # stage 1
        x, H1, W1 = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H1, W1)
        x = self.norm1(x)
        Xb, Hb, Wb = x, H1, W1
        x = x.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H2, W2 = self.patch_embed2(x)
        x, Xb = self.apply_hila(B, x, Xb, H2, W2, H1, W1, '12', self.norm2, self.block2)
        outs[-1] = Xb
        Xb, Hb, Wb = x, H2, W2
        x = x.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H3, W3 = self.patch_embed3(x)
        x, Xb = self.apply_hila(B, x, Xb, H3, W3, H2, W2, '23', self.norm3, self.block3)
        outs[-1] = Xb
        Xb, Hb, Wb = x, H3, W3
        x = x.reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H4, W4 = self.patch_embed4(x)
        x, Xb = self.apply_hila(B, x, Xb, H4, W4, H3, W3, '34', self.norm4, self.block4)
        outs[-1] = Xb
        if self.imagenet_pretraining:
            return x.mean(dim=1)

        x = x.reshape(B, H4, W4, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return outs

    def apply_hila(self, B, Xt, Xb, Ht, Wt, Hb, Wb, idx, top_norm, top_sa):
        i, l, k = 0, 0, 0
        for j, blk in enumerate(top_sa):
            if j % self.hila_stride[idx] == 0 and idx in self.bottomup:
                Xt, Xb = self.bottomup[idx][l](Xt, Xb, Ht, Wt, Hb, Wb)
                l = l + 1 if not self.share_bu_weights else l

            Xt = blk(Xt, Ht, Wt)
            if j != (len(top_sa) - 1) and (j - 1) % self.hila_stride[idx] == 0 and idx in self.topdown:
                Xt, Xb = self.topdown[idx][i](Xt, Xb, Ht, Wt, Hb, Wb)
                Xb = self.prevblock[idx][k](Xb, Hb, Wb)
                i = i + 1 if not self.share_td_weights else i
                k = k + 1 if not self.share_bottom_weights else k

            if self.store_attn and idx in self.bottomup:
                if self.topdown[idx][i].topdown_attn.attn is not None:
                    self.attn[idx]['top'].append(self.topdown[idx][i].topdown_attn.attn)
                if self.bottomup[idx][i].bottomup_attn.attn is not None:
                    self.attn[idx]['bottom'].append(self.bottomup[idx][i].bottomup_attn.attn)

        Xt = top_norm(Xt)
        if idx in self.prevnorm:
            Xb = self.prevnorm[idx](Xb)
        Xb = Xb.reshape(B, Hb, Wb, -1).permute(0, 3, 1, 2).contiguous()
        return Xt, Xb

    def forward(self, x):
        x = self.forward_features(x)

        return x


