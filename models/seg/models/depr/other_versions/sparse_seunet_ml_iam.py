# import torch
# from torch import nn
# import importlib.util

# from ..heads.instance_head import InstanceBranch, PriorInstanceBranch, GroupInstanceBranch
# from ..heads.mask_head import MaskBranch

# from models.seg.modules.mixup import MixUpScaler
# from models.seg.models.base import SparseSEUnet as BaseModel
# from models.seg.models.base import DoubleConv, SE_block

# from configs import cfg




# class SparseSEUnet(BaseModel):
#     def __init__(
#         self,
#         cfg: cfg,
#         n_filters=64,
#         pyramid_pooling=True,
#         n_pp_features=144,
#     ):
#         super(SparseSEUnet, self).__init__(cfg, n_filters, pyramid_pooling, n_pp_features)  
        
#         self.coord_conv = cfg.model.coord_conv
#         self.multi_level = cfg.model.multi_level
#         self.kernel_dim = cfg.model.kernel_dim
#         self.num_masks = cfg.model.num_masks
#         self.num_convs = cfg.model.num_convs
        
#         # mask branch.
#         self.mask_branch = nn.ModuleList([])
#         # self.mask_branch.append(MaskBranch(208))
#         for i in range(self.n_levels):
#             if i == 0:
#                 self.mask_branch.append(MaskBranch(208, num_convs=self.num_convs))
#             else:
#                 self.mask_branch.append(MaskBranch(208+128, num_convs=self.num_convs))
        
#         # instance features.
#         self.prior_instance_branch = nn.ModuleList([])
#         # self.prior_instance_branch.append(PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4))
#         for i in range(self.n_levels):
#             if i == 0:
#                 self.prior_instance_branch.append(PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=self.num_convs))
#             else:
#                 self.prior_instance_branch.append(PriorInstanceBranch(in_channels=464, out_channels=256, num_convs=self.num_convs))

#         # instance branch.
#         self.instance_branch = nn.ModuleList([])
#         for i in range(self.n_levels):
#             if i == 0:
#                 self.instance_branch.append(InstanceBranch(dim=256, kernel_dim=self.kernel_dim, num_masks=self.num_masks))
#             else:
#                 self.instance_branch.append(InstanceBranch(dim=256+self.num_masks, kernel_dim=self.kernel_dim, num_masks=self.num_masks))
        
        

#     # TESTING: add instance and mask branches only to the final layer of the decoder
#     def forward(self, x, idx=None):
#         down_conv_out_tensors = []
#         down_pp_out_tensors = []
#         down_pool_out_tensors = []
        
#         # go down
#         for i in range(self.n_levels):
#             x = self.down_conv_layers[i](x)
#             x = self.down_se_blocks[i](x)
#             down_conv_out_tensors.append(x)
#             if self.pyramid_pooling:
#                 x_pp = self.down_pp_layers[i](x)
#                 x_pp = self.pp_se_blocks[i](x_pp)
#                 down_pp_out_tensors.append(x_pp)
#             x = nn.MaxPool2d(2)(x)
            
#             down_pool_out_tensors.append(x)
#             # Skip connection if required
#             if i > 0:
#                 x = nn.MaxPool2d(2)(down_pool_out_tensors[-2])
#                 x = torch.cat([x, down_pool_out_tensors[-1]], dim=1)
                
#         # middle
#         x = self.middleConv(x)
#         x = self.middleSE(x)
        
#         # go up
#         def go_up(x):
#             total_iam = {}

#             for i in range(self.n_levels):
#                 # if self.coord_conv:
#                 coord_features = self.compute_coordinates(x)
#                 x = torch.cat([coord_features, x], dim=1)
                
#                 x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
#                 x = self.up_conv_layers[i](x)
#                 x = self.up_se_blocks[i](x)
                
#                 if self.pyramid_pooling:
#                     x = torch.cat([x, down_pp_out_tensors[-(i + 1)]], dim=1)
#                 else:
#                     x = torch.cat([x, down_conv_out_tensors[-(i + 1)]], dim=1)
                
                
#                 # multi-level
#                 # if self.multi_level:
#                 if i != 0:
#                     mb = nn.UpsamplingBilinear2d(scale_factor=2)(mb)    # (1, 128, 128, 128)
#                     mb = torch.cat([x, mb], dim=1)
#                     mb = self.mask_branch[i](mb)     
#                 else:
#                     mb = self.mask_branch[i](x)

#                 if i != 0:
#                     # scale: (B, N, Hx, Wx) -> (B, N, Hx * 2, Wx * 2)
#                     # x features shape: (B, Di, Hx * 2, Wx * 2)
#                     inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
#                     inst_feats = torch.cat([x, inst_feats], dim=1)
#                     inst_feats = self.prior_instance_branch[i](inst_feats)
#                 else:
#                     # inst_feats shape: (B, Dm, Hx, Wx)
#                     inst_feats = self.prior_instance_branch[i](x)
                    
#                 # single-level
#                 # else:
#                 # if i == self.n_levels - 1:
#                 #     mb = self.mask_branch[0](x)
#                 #     inst_feats = self.prior_instance_branch[0](x)
                

#                 if i != 0:
#                     _iam = iam["iam"]
#                     _iam = nn.UpsamplingBilinear2d(scale_factor=2)(_iam)
#                     _inst_feats = torch.cat([inst_feats, _iam], dim=1)
#                     logits, kernel, scores, iam = self.instance_branch[i](_inst_feats, idx)
#                 else:
#                     logits, kernel, scores, iam = self.instance_branch[i](inst_feats, idx)

#                 for am in iam:
#                     total_iam[f'{am}_layer_{i}'] = iam[am]

#                 # if i == self.n_levels - 1:
#                 #     logits, kernel, scores, iam = self.instance_branch[i](inst_feats, idx)

#             return x, mb, (logits, kernel, scores, total_iam)
    
#         # cyto
#         x, mask_features, (logits, kernel, scores, iam) = go_up(x)
        
#         # Predicting instance masks
#         N = kernel.shape[1]  # num_masks
#         B, C, H, W = mask_features.shape

#         masks = torch.bmm(
#             kernel,    # (B, N, 128)
#             mask_features.view(B, C, H * W)   # (B, 128, [HW])
#         ) # -> (B, N, [HW])
#         masks = masks.view(B, N, H, W)  # (B, N, [HW]) -> (B, N, H, W)
        
#         output = {
#             'pred_logits': logits,
#             'pred_scores': scores,
#             'pred_iam': iam,
#             'pred_masks': masks,
#             'pred_kernel': kernel,
#         }
    
#         return output




import torch
from torch import nn

from ..heads.instance_head import InstanceBranch, PriorInstanceBranch, GroupInstanceBranch
from ..heads.mask_head import MaskBranch

from models.seg.modules.mixup import MixUpScaler
from models.seg.models.base import SparseSEUnet as BaseModel
from models.seg.models.base import DoubleConv, SE_block
from models.seg.blocks import ContextBlock

from configs import cfg


class SparseSEUnet(BaseModel):
    def __init__(
        self,
        cfg: cfg,
        n_filters=64,
        pyramid_pooling=True,
        n_pp_features=144,
    ):
        super(SparseSEUnet, self).__init__(cfg, n_filters, pyramid_pooling, n_pp_features)  
        
        self.coord_conv = cfg.model.coord_conv
        self.multi_level = cfg.model.multi_level
        self.kernel_dim = cfg.model.kernel_dim
        self.num_masks = cfg.model.num_masks
        self.num_convs = cfg.model.num_convs
        
        # mb.
        self.up_se_blocks_mb = nn.ModuleList([])
        self.up_conv_layers_mb = nn.ModuleList([])
        for _ in range(self.n_levels):
            # up convolution
            if len(self.up_conv_layers_mb) == 0:
                # if self.coord_conv:
                upconv = DoubleConv(self.n_filters+2, self.n_filters)
                # else:
                #     upconv = DoubleConv(self.n_filters, self.n_filters)
            else:
                # if self.coord_conv:
                upconv = DoubleConv(
                    (self.n_filters // 4) * 5 + 2 * self.n_filters+2, self.n_filters
                )
                # else:
                #     upconv = DoubleConv(
                #         (self.n_filters // 4) * 5 + 2 * self.n_filters, self.n_filters
                #     )
            self.up_conv_layers_mb.append(upconv)

             # SE blocks following the upconv 
            up_se = SE_block(num_features=self.n_filters)            
            self.up_se_blocks_mb.append(up_se)


        

        # mask branch.
        self.mask_branch = nn.ModuleList([])
        # self.mask_branch.append(MaskBranch(208))
        for i in range(self.n_levels):
            if i == 0:
                self.mask_branch.append(MaskBranch(208, num_convs=self.num_convs))
            else:
                self.mask_branch.append(MaskBranch(208+128, num_convs=self.num_convs))
        
        # instance features.
        self.prior_instance_branch = nn.ModuleList([])
        # self.prior_instance_branch.append(PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4))
        for i in range(self.n_levels):
            if i == 0:
                self.prior_instance_branch.append(PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=self.num_convs))
            else:
                self.prior_instance_branch.append(PriorInstanceBranch(in_channels=464, out_channels=256, num_convs=self.num_convs))

        # instance branch.
        self.instance_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                self.instance_branch.append(InstanceBranch(dim=256, kernel_dim=self.kernel_dim, num_masks=self.num_masks))
            else:
                self.instance_branch.append(InstanceBranch(dim=256+self.num_masks, kernel_dim=self.kernel_dim, num_masks=self.num_masks))
        
        

    def forward(self, x, idx=None):
        down_conv_out_tensors = []
        down_pp_out_tensors = []
        down_pool_out_tensors = []
        
        # go down
        for i in range(self.n_levels):
            x = self.down_conv_layers[i](x)
            x = self.down_se_blocks[i](x)
            down_conv_out_tensors.append(x)
            if self.pyramid_pooling:
                x_pp = self.down_pp_layers[i](x)
                x_pp = self.pp_se_blocks[i](x_pp)
                down_pp_out_tensors.append(x_pp)
            x = nn.MaxPool2d(2)(x)
            
            down_pool_out_tensors.append(x)
            # Skip connection if required
            if i > 0:
                x = nn.MaxPool2d(2)(down_pool_out_tensors[-2])
                x = torch.cat([x, down_pool_out_tensors[-1]], dim=1)
                
        # middle
        x = self.middleConv(x)
        x = self.middleSE(x)
        
        # go up
        def go_up(x):
            total_iam = {}
            x_mb = x.clone()

            for i in range(self.n_levels):
                # if self.coord_conv:
                coord_features = self.compute_coordinates(x)
                x = torch.cat([coord_features, x], dim=1)
                x_mb = torch.cat([coord_features, x_mb], dim=1)
                
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
                x = self.up_conv_layers[i](x)
                x = self.up_se_blocks[i](x)

                x_mb = nn.UpsamplingBilinear2d(scale_factor=2)(x_mb)
                x_mb = self.up_conv_layers_mb[i](x_mb)
                x_mb = self.up_se_blocks_mb[i](x_mb)


                if self.pyramid_pooling:
                    x = torch.cat([x, down_pp_out_tensors[-(i + 1)]], dim=1)
                    x_mb = torch.cat([x_mb, down_pp_out_tensors[-(i + 1)]], dim=1)
                else:
                    x = torch.cat([x, down_conv_out_tensors[-(i + 1)]], dim=1)
                    x_mb = torch.cat([x_mb, down_conv_out_tensors[-(i + 1)]], dim=1)
                
                # multi-level
                # if self.multi_level:
                if i != 0:
                    mb = nn.UpsamplingBilinear2d(scale_factor=2)(mb)    # (1, 128, 128, 128)
                    mb = torch.cat([x_mb, mb], dim=1)
                    mb = self.mask_branch[i](mb)     
                else:
                    mb = self.mask_branch[i](x_mb)

                if i != 0:
                    # scale: (B, N, Hx, Wx) -> (B, N, Hx * 2, Wx * 2)
                    # x features shape: (B, Di, Hx * 2, Wx * 2)
                    inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
                    inst_feats = torch.cat([x, inst_feats], dim=1)
                    inst_feats = self.prior_instance_branch[i](inst_feats)
                else:
                    # inst_feats shape: (B, Dm, Hx, Wx)
                    inst_feats = self.prior_instance_branch[i](x)

                if i != 0:
                    _iam = iam["iam"]
                    _iam = nn.UpsamplingBilinear2d(scale_factor=2)(_iam)
                    _inst_feats = torch.cat([inst_feats, _iam], dim=1)
                    logits, kernel, scores, iam = self.instance_branch[i](_inst_feats, idx)
                else:
                    logits, kernel, scores, iam = self.instance_branch[i](inst_feats, idx)

                for am in iam:
                    total_iam[f'{am}_layer_{i}'] = iam[am]

                # if i == self.n_levels - 1:
                #     logits, kernel, scores, iam = self.instance_branch[i](inst_feats, idx)

            return x, mb, (logits, kernel, scores, total_iam)
    
        # cyto
        x, mask_features, (logits, kernel, scores, iam) = go_up(x)
        
        # Predicting instance masks
        N = kernel.shape[1]  # num_masks
        B, C, H, W = mask_features.shape

        masks = torch.bmm(
            kernel,    # (B, N, 128)
            mask_features.view(B, C, H * W)   # (B, 128, [HW])
        ) # -> (B, N, [HW])
        masks = masks.view(B, N, H, W)  # (B, N, [HW]) -> (B, N, H, W)
        
        output = {
            'pred_logits': logits,
            'pred_scores': scores,
            'pred_iam': iam,
            'pred_masks': masks,
            'pred_kernel': kernel,
        }
    
        return output
