import torch
from torch import nn
from fvcore.nn.weight_init import c2_msra_fill

import sys
sys.path.append("./")

from models.seg.heads.instance_head import InstanceBranch, PriorInstanceBranch, GroupInstanceBranch
from models.seg.heads.mask_head import MaskBranch

from models.seg.modules.mixup import MixUpScaler
from models.seg.models.base import SparseSEUnet as BaseModel
from models.seg.models.base import DoubleConv, SE_block
from models.seg.blocks import ContextBlock

from configs import cfg
from utils.registry import MODELS


@MODELS.register(name="sparse_seunet_occluder")
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
        
        # instance.
        self.up_conv_layers = nn.ModuleList([])
        for _ in range(self.n_levels):
            if len(self.up_conv_layers) == 0:
                upconv = DoubleConv(self.n_filters + 2, self.n_filters)
            else:
                upconv = DoubleConv(
                    (self.n_filters // 4) * 5 + 2 * self.n_filters + 2, self.n_filters
                )
            self.up_conv_layers.append(upconv)


        # occluders.
        self.up_se_blocks_occluders = nn.ModuleList([])
        self.up_conv_layers_occluders = nn.ModuleList([])
        for _ in range(self.n_levels):
            # up convolution
            if len(self.up_conv_layers_occluders) == 0:
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
            self.up_conv_layers_occluders.append(upconv)

             # SE blocks following the upconv 
            up_se = SE_block(num_features=self.n_filters)            
            self.up_se_blocks_occluders.append(up_se)


        # mask branch.
        self.mask_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                self.mask_branch.append(MaskBranch(208))
            else:
                self.mask_branch.append(MaskBranch(464))
        
        # instance features.
        self.prior_instance_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                self.prior_instance_branch.append(PriorInstanceBranch(in_channels=208))
            # elif i == self.n_levels - 1:
            #     self.prior_instance_branch.append(
            #         # nn.ModuleList([PriorInstanceBranch(in_channels=464)])
            #         PriorInstanceBranch(in_channels=720)
            #         )
            else:
                self.prior_instance_branch.append(PriorInstanceBranch(in_channels=464))


        # occluder.
        # mask branch.
        self.mask_branch_occluder = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                self.mask_branch_occluder.append(MaskBranch(208))
            else:
                self.mask_branch_occluder.append(MaskBranch(464))
        
        # instance features.
        self.prior_occluder_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                self.prior_occluder_branch.append(PriorInstanceBranch(in_channels=208))
            else:
                self.prior_occluder_branch.append(PriorInstanceBranch(in_channels=464))

        # self.occluder_context_block = ContextBlock(inplanes=464, ratio=4, pooling_type='att', fusion_types=['channel_add', 'channel_mul'])
        # self.instance_context_block = ContextBlock(inplanes=464, ratio=4, pooling_type='att', fusion_types=['channel_add', 'channel_mul'])
        # self.out_occluder_branch = PriorInstanceBranch(in_channels=464, out_channels=256)
        # self.out_instance_branch = PriorInstanceBranch(in_channels=464, out_channels=256)

        # instance branch.
        # self.instance_branch = InstanceBranch(kernel_dim=self.kernel_dim, num_masks=self.num_masks)
        self.occluder_branch = InstanceBranch(
            dim=256, 
            kernel_dim=self.kernel_dim, 
            num_masks=self.num_masks
            )
        # self.occluder_bound_branch = GroupInstanceBranch(
        #     dim=256, 
        #     kernel_dim=self.kernel_dim, 
        #     num_masks=self.num_masks
        #     )
        
        self.instance_branch = InstanceBranch(
            dim=256, 
            kernel_dim=self.kernel_dim, 
            num_masks=self.num_masks
            )
        # self.instance_bound_branch = GroupInstanceBranch(
        #     dim=256, 
        #     kernel_dim=self.kernel_dim, 
        #     num_masks=self.num_masks
        #     )


    # def init_weights(self):
    #     module_mapping = {
    #         "up_se_blocks": "up_se_blocks_occluder",
    #         "up_conv_layers": "up_conv_layers_occluder",
    #         "mask_branch_occluder": "mask_branch",
    #         "prior_instance_branch": "prior_occluder_branch",
    #         "instance_branch": "occluder_branch"
    #     }
        

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
            x_occl = x.clone()

            for i in range(self.n_levels):
                # if self.coord_conv:
                coord_features = self.compute_coordinates(x)
                x = torch.cat([coord_features, x], dim=1)
                x_occl = torch.cat([coord_features, x_occl], dim=1)
                
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
                x = self.up_conv_layers[i](x)
                x = self.up_se_blocks[i](x)

                x_occl = nn.UpsamplingBilinear2d(scale_factor=2)(x_occl)
                x_occl = self.up_conv_layers_occluders[i](x_occl)
                x_occl = self.up_se_blocks_occluders[i](x_occl)


                if self.pyramid_pooling:
                    x = torch.cat([x, down_pp_out_tensors[-(i + 1)]], dim=1)
                    x_occl = torch.cat([x_occl, down_pp_out_tensors[-(i + 1)]], dim=1)
                else:
                    x = torch.cat([x, down_conv_out_tensors[-(i + 1)]], dim=1)
                    x_occl = torch.cat([x_occl, down_conv_out_tensors[-(i + 1)]], dim=1)
                
                # multi-level
                # if self.multi_level:
                if i != 0:
                    mb_inst = nn.UpsamplingBilinear2d(scale_factor=2)(mb_inst)    # (1, 128, 128, 128)
                    mb_inst = torch.cat([x, mb_inst], dim=1)
                    mb_inst = self.mask_branch[i](mb_inst)     

                    mb_occl = nn.UpsamplingBilinear2d(scale_factor=2)(mb_occl)    # (1, 128, 128, 128)
                    mb_occl = torch.cat([x_occl, mb_occl], dim=1)
                    mb_occl = self.mask_branch_occluder[i](mb_occl)    
                else:
                    mb_inst = self.mask_branch[i](x)
                    mb_occl = self.mask_branch_occluder[i](x_occl)

                if i != 0:
                    # scale: (B, N, Hx, Wx) -> (B, N, Hx * 2, Wx * 2)
                    # x features shape: (B, Di, Hx * 2, Wx * 2)
                    inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
                    inst_feats = torch.cat([x, inst_feats], dim=1)

                    occl_feats = nn.UpsamplingBilinear2d(scale_factor=2)(occl_feats)
                    occl_feats = torch.cat([x_occl, occl_feats], dim=1)

                    if i != self.n_levels - 1:
                        inst_feats = self.prior_instance_branch[i](inst_feats)
                        occl_feats = self.prior_occluder_branch[i](occl_feats)
                else:
                    # inst_feats shape: (B, Dm, Hx, Wx)
                    inst_feats = self.prior_instance_branch[i](x)
                    occl_feats = self.prior_occluder_branch[i](x_occl)

                    
                # single-level
                # else:
                # if i == self.n_levels - 1:
                #     mb = self.mask_branch[0](x)
                #     inst_feats = self.prior_instance_branch[0](x)
                        
                if i == self.n_levels - 1:
                    # occl_feats = inst_feats.clone()
                    # occl_feats = self.occluder_context_block(occl_feats)
                    occl_feats = self.prior_occluder_branch[i](occl_feats)  # (256, H, W)

                    # logits, occl_bound_kernel, scores, occl_bound_iam = self.occluder_bound_branch(occl_feats)
                    occl_logits, occl_kernel, occl_scores, occl_iam, occl_coords = self.occluder_branch(occl_feats)

                    # inst_feats = inst_feats + occl_feats
                    # inst_feats
                    # inst_feats = inst_feats + occl_feats # (256, H, W)
                    # inst_feats = torch.cat([inst_feats, occl_feats], dim=1)
                    # inst_feats = self.instance_context_block(inst_feats)
                    inst_feats = self.prior_instance_branch[i](inst_feats) # (256, H, W)
                    # inst_feats = inst_feats + occl_feats # (256, H, W)

                    # logits, inst_bound_kernel, scores, inst_bound_iam = self.instance_bound_branch(inst_feats)
                    inst_logits, inst_kernel, inst_scores, inst_iam, inst_coords = self.instance_branch(inst_feats)

                    
                    mb = {
                        "mb_occluder": mb_occl,
                        "mb_instance": mb_inst
                    }

                    logits = {
                        "occluder_logits": occl_logits,
                        "instance_logits": inst_logits,
                    }
                    scores = {
                        "occluder_scores": occl_scores,
                        "instance_scores": inst_scores,
                    }
                    kernel = {
                        "occluder_kernel": occl_kernel,
                        # "occluder_bound_kernel": occl_bound_kernel,
                        "instance_kernel": inst_kernel,
                        # "instance_bound_kernel": inst_bound_kernel
                    }

                    iam = {
                        "occluder_iam": occl_iam,
                        # "occluder_bound_iam": occl_bound_iam,
                        "instance_iam": inst_iam,
                        # "instance_bound_iam": inst_bound_iam
                    }

                    coords = {
                        "occluder_coords": occl_coords,
                        "instance_coords": inst_coords,
                    }

            return x, mb, (logits, kernel, scores, iam, coords)
    
        # cyto
        x, mask_features, (logits, kernel, scores, iam, coords) = go_up(x)

        mask_features_inst = mask_features["mb_instance"]
        mask_features_occl = mask_features["mb_occluder"]
        
        inst_kernel = kernel["instance_kernel"]
        # inst_bound_kernel = kernel["instance_bound_kernel"]
        occl_kernel = kernel["occluder_kernel"]
        # occl_bound_kernel = kernel["occluder_bound_kernel"]

        # Predicting instance masks
        N = inst_kernel.shape[1]  # num_masks
        B, C, H, W = mask_features_inst.shape

        masks = torch.bmm(
            inst_kernel,    # (B, N, 128)
            mask_features_inst.view(B, C, H * W)   # (B, 128, [HW])
        ) # -> (B, N, [HW])
        masks = masks.view(B, N, H, W)  # (B, N, [HW]) -> (B, N, H, W)

        # masks_bounds = torch.bmm(
        #     inst_bound_kernel,    # (B, N, 128)
        #     mask_features.view(B, C, H * W)   # (B, 128, [HW])
        # ) # -> (B, N, [HW])
        # masks_bounds = masks_bounds.view(B, N, H, W)  # (B, N, [HW]) -> (B, N, H, W)

        occluders = torch.bmm(
            occl_kernel,    # (B, N, 128)
            mask_features_occl.view(B, C, H * W)   # (B, 128, [HW])
        ) # -> (B, N, [HW])
        occluders = occluders.view(B, N, H, W)  # (B, N, [HW]) -> (B, N, H, W)

        # occluders_bounds = torch.bmm(
        #     occl_bound_kernel,    # (B, N, 128)
        #     mask_features.view(B, C, H * W)   # (B, 128, [HW])
        # ) # -> (B, N, [HW])
        # occluders_bounds = occluders_bounds.view(B, N, H, W)  # (B, N, [HW]) -> (B, N, H, W)


        inst_iam = iam["instance_iam"]
        occl_iam = iam["occluder_iam"]

        inst_logits = logits["instance_logits"]
        occl_logits = logits["occluder_logits"]

        inst_scores = scores["instance_scores"]
        occl_scores = scores["occluder_scores"]

        inst_coords = coords["instance_coords"]
        occl_coords = coords["occluder_coords"]


        output = {
            'pred_logits': inst_logits,
            'pred_scores': inst_scores,
            'pred_iam': inst_iam,
            'pred_masks': masks,  # instnace masks
            'pred_kernel': kernel,
            'pred_occluders_masks': occluders, # occluders masks
            'pred_occluders_logits': occl_logits,
            'pred_occluders_scores': occl_scores,
            'pred_occluders_iam': occl_iam,

            'pred_coords': inst_coords,
            'pred_occluders_coords': occl_coords
        }
    
        return output


# arch 17.10.23
# class SparseSEUnet(BaseModel):
#     """
#     Base SparseUnet + Simple Instance-Occluder interaction
#     """
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


#         self.up_conv_layers = nn.ModuleList([])
#         # CHANGED: removed additional +2 channels
#         # CHANGED: moved the coordinate features addition to instance branches
#         for _ in range(self.n_levels):
#             # up convolutions
#             if len(self.up_conv_layers) == 0:
#                 upconv = DoubleConv(self.n_filters, self.n_filters)
#             else:
#                 upconv = DoubleConv(
#                     (self.n_filters // 4) * 5 + 2 * self.n_filters, self.n_filters
#                 )
#             self.up_conv_layers.append(upconv)

        
#         # # mask branch.
#         # self.mask_branch = nn.ModuleList([])
#         # for i in range(self.n_levels):
#         #     if i == 0:
#         #         self.mask_branch.append(MaskBranch(in_channels=208 + 2, out_channels=256, num_convs=4))
#         #     else:
#         #         self.mask_branch.append(MaskBranch(in_channels=464 + 2, out_channels=256, num_convs=4))
        
#         # # mask feature map projection layer (1x1)
#         # self.projection = nn.Conv2d(256, self.kernel_dim, kernel_size=1)
        
#         # # instance features.
#         # self.prior_instance_branch = nn.ModuleList([])
#         # for i in range(self.n_levels):
#         #     if i == 0:
#         #         self.prior_instance_branch.append(PriorInstanceBranch(in_channels=208 + 2, out_channels=256, num_convs=self.num_convs))
#         #     else:
#         #         self.prior_instance_branch.append(PriorInstanceBranch(in_channels=464 + 2, out_channels=256, num_convs=self.num_convs))


#         # self.last_instance_branch = nn.ModuleList([
#         #     PriorInstanceBranch(in_channels=256 + 2, out_channels=256 + 2, num_convs=2),
#         #     PriorInstanceBranch(in_channels=256 + 2, out_channels=256 + 2, num_convs=2)
#         # ])
#         # self.last_occluder_branch = nn.ModuleList([
#         #     PriorInstanceBranch(in_channels=256 + 2, out_channels=256 + 2, num_convs=2),
#         #     PriorInstanceBranch(in_channels=256 + 2, out_channels=256 + 2, num_convs=2)
#         # ])

#         # # instance branch.
#         # self.instance_branch = InstanceBranch(dim=256 + 2, kernel_dim=self.kernel_dim, num_masks=self.num_masks)
#         # self.occluder_branch = InstanceBranch(dim=256 + 2, kernel_dim=self.kernel_dim, num_masks=self.num_masks)

#         # mask branch.
#         self.mask_branch = nn.ModuleList([])
#         self.mask_branch.append(MaskBranch(in_channels=208 + 2, out_channels=256, num_convs=4))
        
#         # mask feature map projection layer (1x1)
#         self.projection = nn.Conv2d(256, self.kernel_dim, kernel_size=1)
        
#         # instance features.
#         self.prior_instance_branch = nn.ModuleList([
#             PriorInstanceBranch(in_channels=208 + 2, out_channels=208 + 2, num_convs=2),
#             PriorInstanceBranch(in_channels=208 + 2, out_channels=208 + 2, num_convs=2)
#         ])
#         self.prior_occluder_branch = nn.ModuleList([
#             PriorInstanceBranch(in_channels=208 + 2, out_channels=208 + 2, num_convs=2),
#             PriorInstanceBranch(in_channels=208 + 2, out_channels=208 + 2, num_convs=2)
#         ])

#         # instance branch.
#         self.instance_branch = InstanceBranch(dim=208 + 2, kernel_dim=self.kernel_dim, num_masks=self.num_masks)
#         self.occluder_branch = InstanceBranch(dim=208 + 2, kernel_dim=self.kernel_dim, num_masks=self.num_masks)

#         self.init_weights()

#     def init_weights(self):
#         c2_msra_fill(self.projection)


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
#             for i in range(self.n_levels):

#                 x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
#                 x = self.up_conv_layers[i](x)
#                 x = self.up_se_blocks[i](x)
                
#                 if self.pyramid_pooling:
#                     x = torch.cat([x, down_pp_out_tensors[-(i + 1)]], dim=1)
#                 else:
#                     x = torch.cat([x, down_conv_out_tensors[-(i + 1)]], dim=1)


#                 # coord_features = self.compute_coordinates(x)
#                 # _x = torch.cat([coord_features, x], dim=1)
                
#                 # if i != 0:
#                 #     mb = nn.UpsamplingBilinear2d(scale_factor=2)(mb)
#                 #     mb = torch.cat([_x, mb], dim=1)
#                 #     mb = self.mask_branch[i](mb)     
#                 # else:
#                 #     mb = self.mask_branch[i](_x)

#                 # if i != 0:
#                 #     inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
#                 #     inst_feats = torch.cat([_x, inst_feats], dim=1)
#                 #     inst_feats = self.prior_instance_branch[i](inst_feats)
#                 # else:
#                 #     inst_feats = self.prior_instance_branch[i](_x)
                    

#             # # mask branch.
#             # mb = self.projection(mb)

#             # coord_features = self.compute_coordinates(inst_feats)
#             # inst_feats = torch.cat([coord_features, inst_feats], dim=1)

#             # # occluder branch.
#             # occl_feats = self.last_occluder_branch[0](inst_feats)           # (B, 208+2, H, W) -> (B, 208+2, H, W)
#             # occl_feats = self.last_occluder_branch[1](occl_feats)  # (B, 208+2, H, W) -> (B, 208+2, H, W)

#             # # instance branch.
#             # inst_feats = inst_feats + occl_feats                             # (B, 208+2, H, W)
#             # inst_feats = self.last_instance_branch[0](inst_feats)  # (B, 208+2, H, W) -> (B, 208+2, H, W)
#             # inst_feats = self.last_instance_branch[1](inst_feats)  # (B, 208+2, H, W) -> (B, 208+2, H, W)

#             # inst_logits, inst_kernel, inst_scores, inst_iam = self.instance_branch(inst_feats, idx)
#             # occl_logits, occl_kernel, occl_scores, occl_iam = self.occluder_branch(occl_feats, idx)

#             # last layer
#             coord_features = self.compute_coordinates(x)
#             x = torch.cat([coord_features, x], dim=1)

#             # mask branch.
#             mb = self.mask_branch[0](x)
#             mb = self.projection(mb)


#             # occluder branch.
#             occl_feats = self.prior_occluder_branch[0](x)           # (B, 208+2, H, W) -> (B, 208+2, H, W)
#             occl_feats = self.prior_occluder_branch[1](occl_feats)  # (B, 208+2, H, W) -> (B, 208+2, H, W)

#             # instance branch.
#             inst_feats = x
#             inst_feats = inst_feats + occl_feats                             # (B, 208+2, H, W)
#             inst_feats = self.prior_instance_branch[0](x)  # (B, 208+2, H, W) -> (B, 208+2, H, W)
#             inst_feats = self.prior_instance_branch[1](inst_feats)  # (B, 208+2, H, W) -> (B, 208+2, H, W)
#             # inst_feats = inst_feats + occl_feats                             # (B, 208+2, H, W)


#             # coord_features = self.compute_coordinates(inst_feats)
#             # inst_feats = torch.cat([coord_features, inst_feats], dim=1)

#             # coord_features = self.compute_coordinates(occl_feats)
#             # occl_feats = torch.cat([coord_features, occl_feats], dim=1)

#             inst_logits, inst_kernel, inst_scores, inst_iam = self.instance_branch(inst_feats, idx)
#             occl_logits, occl_kernel, occl_scores, occl_iam = self.occluder_branch(occl_feats, idx)
            

#             mb = {
#                 "mb": mb,
#             }
            
#             logits = {
#                 "occluder_logits": occl_logits,
#                 "instance_logits": inst_logits,
#             }
#             scores = {
#                 "occluder_scores": occl_scores,
#                 "instance_scores": inst_scores,
#             }

#             kernel = {
#                 "occluder_kernel": occl_kernel,
#                 "instance_kernel": inst_kernel,
#             }

#             iam = {
#                 "occluder_iam": occl_iam,
#                 "instance_iam": inst_iam,
#             }

#             return x, mb, (logits, kernel, scores, iam)
    
#         # cyto
#         x, mask_features, (logits, kernel, scores, iam) = go_up(x)
        
#         mask_features = mask_features["mb"]
        
#         inst_kernel = kernel["instance_kernel"]
#         occl_kernel = kernel["occluder_kernel"]

#         # Predicting instance masks
#         N = inst_kernel.shape[1]  # num_masks
#         B, C, H, W = mask_features.shape

#         masks = torch.bmm(
#             inst_kernel, 
#             mask_features.view(B, C, H * W)
#         ) 
#         masks = masks.view(B, N, H, W) 

#         occluders = torch.bmm(
#             occl_kernel,    
#             mask_features.view(B, C, H * W)   
#         ) 
#         occluders = occluders.view(B, N, H, W) 


#         inst_iam = iam["instance_iam"]
#         occl_iam = iam["occluder_iam"]

#         inst_logits = logits["instance_logits"]
#         occl_logits = logits["occluder_logits"]

#         inst_scores = scores["instance_scores"]
#         occl_scores = scores["occluder_scores"]


#         output = {
#             'pred_logits': inst_logits,
#             'pred_scores': inst_scores,
#             'pred_iam': inst_iam,
#             'pred_masks': masks,  # instnace masks
#             'pred_kernel': kernel,
#             'pred_occluders_masks': occluders, # occluders masks
#             'pred_occluders_logits': occl_logits,
#             'pred_occluders_scores': occl_scores,
#             'pred_occluders_iam': occl_iam
#         }
    
#         return output
    


if __name__ == "__main__":
    model = SparseSEUnet(cfg)
    x = torch.rand(2, 2, 512, 512)
    out = model(x)
    print(out["pred_masks"].shape)


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
        
#         # occluders.
#         self.up_se_blocks_occluders = nn.ModuleList([])
#         self.up_conv_layers_occluders = nn.ModuleList([])
#         for _ in range(self.n_levels):
#             # up convolution
#             if len(self.up_conv_layers_occluders) == 0:
#                 # if self.coord_conv:
#                 upconv = DoubleConv(self.n_filters+2, self.n_filters)
#                 # else:
#                 #     upconv = DoubleConv(self.n_filters, self.n_filters)
#             else:
#                 # if self.coord_conv:
#                 upconv = DoubleConv(
#                     (self.n_filters // 4) * 5 + 2 * self.n_filters+2, self.n_filters
#                 )
#                 # else:
#                 #     upconv = DoubleConv(
#                 #         (self.n_filters // 4) * 5 + 2 * self.n_filters, self.n_filters
#                 #     )
#             self.up_conv_layers_occluders.append(upconv)

#              # SE blocks following the upconv 
#             up_se = SE_block(num_features=self.n_filters)            
#             self.up_se_blocks_occluders.append(up_se)


#         # mask branch.
#         self.mask_branch = nn.ModuleList([])
#         for i in range(self.n_levels):
#             if i == 0:
#                 self.mask_branch.append(MaskBranch(208))
#             else:
#                 self.mask_branch.append(MaskBranch(208+128))
        
#         # instance features.
#         self.prior_instance_branch = nn.ModuleList([])
#         for i in range(self.n_levels):
#             if i == 0:
#                 self.prior_instance_branch.append(PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4))
#             else:
#                 self.prior_instance_branch.append(PriorInstanceBranch(in_channels=464, out_channels=256, num_convs=4))


#         # occluder.
#         # mask branch.
#         self.mask_branch_occluder = nn.ModuleList([])
#         for i in range(self.n_levels):
#             if i == 0:
#                 self.mask_branch_occluder.append(MaskBranch(208))
#             else:
#                 self.mask_branch_occluder.append(MaskBranch(208+128))
        
#         # instance features.
#         self.prior_occluder_branch = nn.ModuleList([])
#         for i in range(self.n_levels):
#             if i == 0:
#                 self.prior_occluder_branch.append(PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4))
#             else:
#                 self.prior_occluder_branch.append(PriorInstanceBranch(in_channels=464, out_channels=256, num_convs=4))

#         # self.occluder_context_block = ContextBlock(inplanes=464, ratio=4, pooling_type='att', fusion_types=['channel_add', 'channel_mul'])
#         # self.instance_context_block = ContextBlock(inplanes=464, ratio=4, pooling_type='att', fusion_types=['channel_add', 'channel_mul'])
#         # self.out_occluder_branch = PriorInstanceBranch(in_channels=464, out_channels=256)
#         # self.out_instance_branch = PriorInstanceBranch(in_channels=464, out_channels=256)

#         # instance branch.
#         # self.instance_branch = InstanceBranch(kernel_dim=self.kernel_dim, num_masks=self.num_masks)
#         self.occluder_branch = GroupInstanceBranch(
#             dim=256, 
#             kernel_dim=self.kernel_dim, 
#             num_masks=self.num_masks
#             )
#         # self.occluder_bound_branch = GroupInstanceBranch(
#         #     dim=256, 
#         #     kernel_dim=self.kernel_dim, 
#         #     num_masks=self.num_masks
#         #     )
        
#         self.instance_branch = GroupInstanceBranch(
#             dim=256, 
#             kernel_dim=self.kernel_dim, 
#             num_masks=self.num_masks
#             )
#         # self.instance_bound_branch = GroupInstanceBranch(
#         #     dim=256, 
#         #     kernel_dim=self.kernel_dim, 
#         #     num_masks=self.num_masks
#         #     )


#     # def init_weights(self):
#     #     module_mapping = {
#     #         "up_se_blocks": "up_se_blocks_occluder",
#     #         "up_conv_layers": "up_conv_layers_occluder",
#     #         "mask_branch_occluder": "mask_branch",
#     #         "prior_instance_branch": "prior_occluder_branch",
#     #         "instance_branch": "occluder_branch"
#     #     }
        

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
#             x_occl = x.clone()

#             for i in range(self.n_levels):
#                 # if self.coord_conv:
#                 coord_features = self.compute_coordinates(x)
#                 x = torch.cat([coord_features, x], dim=1)
#                 x_occl = torch.cat([coord_features, x_occl], dim=1)
                
#                 x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
#                 x = self.up_conv_layers[i](x)
#                 x = self.up_se_blocks[i](x)

#                 x_occl = nn.UpsamplingBilinear2d(scale_factor=2)(x_occl)
#                 x_occl = self.up_conv_layers_occluders[i](x_occl)
#                 x_occl = self.up_se_blocks_occluders[i](x_occl)


#                 if self.pyramid_pooling:
#                     x = torch.cat([x, down_pp_out_tensors[-(i + 1)]], dim=1)
#                     x_occl = torch.cat([x_occl, down_pp_out_tensors[-(i + 1)]], dim=1)
#                 else:
#                     x = torch.cat([x, down_conv_out_tensors[-(i + 1)]], dim=1)
#                     x_occl = torch.cat([x_occl, down_conv_out_tensors[-(i + 1)]], dim=1)
                
#                 # multi-level
#                 # if self.multi_level:
#                 if i != 0:
#                     mb_inst = nn.UpsamplingBilinear2d(scale_factor=2)(mb_inst)    # (1, 128, 128, 128)
#                     mb_inst = torch.cat([x, mb_inst], dim=1)
#                     mb_inst = self.mask_branch[i](mb_inst)     

#                     mb_occl = nn.UpsamplingBilinear2d(scale_factor=2)(mb_occl)    # (1, 128, 128, 128)
#                     mb_occl = torch.cat([x_occl, mb_occl], dim=1)
#                     mb_occl = self.mask_branch_occluder[i](mb_occl)    
#                 else:
#                     mb_inst = self.mask_branch[i](x)
#                     mb_occl = self.mask_branch_occluder[i](x_occl)

#                 if i != 0:
#                     # scale: (B, N, Hx, Wx) -> (B, N, Hx * 2, Wx * 2)
#                     # x features shape: (B, Di, Hx * 2, Wx * 2)
#                     inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
#                     inst_feats = torch.cat([x, inst_feats], dim=1)

#                     occl_feats = nn.UpsamplingBilinear2d(scale_factor=2)(occl_feats)
#                     occl_feats = torch.cat([x_occl, occl_feats], dim=1)

#                     if i != self.n_levels - 1:
#                         inst_feats = self.prior_instance_branch[i](inst_feats)
#                         occl_feats = self.prior_occluder_branch[i](occl_feats)
#                 else:
#                     # inst_feats shape: (B, Dm, Hx, Wx)
#                     inst_feats = self.prior_instance_branch[i](x)
#                     occl_feats = self.prior_occluder_branch[i](x_occl)

                    
#                 # single-level
#                 # else:
#                 # if i == self.n_levels - 1:
#                 #     mb = self.mask_branch[0](x)
#                 #     inst_feats = self.prior_instance_branch[0](x)
                        
#                 if i == self.n_levels - 1:
#                     # occl_feats = inst_feats.clone()
#                     # occl_feats = self.occluder_context_block(occl_feats)
#                     occl_feats = self.prior_occluder_branch[i](occl_feats)  # (256, H, W)

#                     # logits, occl_bound_kernel, scores, occl_bound_iam = self.occluder_bound_branch(occl_feats)
#                     occl_logits, occl_kernel, occl_scores, occl_iam = self.occluder_branch(occl_feats)

#                     # inst_feats = inst_feats + occl_feats
#                     # inst_feats
#                     # inst_feats = self.instance_context_block(inst_feats)
#                     inst_feats = self.prior_instance_branch[i](inst_feats) # (256, H, W)
#                     # inst_feats = inst_feats + occl_feats # (256, H, W)

#                     # logits, inst_bound_kernel, scores, inst_bound_iam = self.instance_bound_branch(inst_feats)
#                     inst_logits, inst_kernel, inst_scores, inst_iam = self.instance_branch(inst_feats)

                    
#                     mb = {
#                         "mb_occluder": mb_occl,
#                         "mb_instance": mb_inst
#                     }
                    
#                     logits = {
#                         "occluder_logits": occl_logits,
#                         "instance_logits": inst_logits,
#                     }
#                     scores = {
#                         "occluder_scores": occl_scores,
#                         "instance_scores": inst_scores,
#                     }

#                     kernel = {
#                         "occluder_kernel": occl_kernel,
#                         # "occluder_bound_kernel": occl_bound_kernel,
#                         "instance_kernel": inst_kernel,
#                         # "instance_bound_kernel": inst_bound_kernel
#                     }

#                     iam = {
#                         "occluder_iam": occl_iam,
#                         # "occluder_bound_iam": occl_bound_iam,
#                         "instance_iam": inst_iam,
#                         # "instance_bound_iam": inst_bound_iam
#                     }

#             return x, mb, (logits, kernel, scores, iam)
    
#         # cyto
#         x, mask_features, (logits, kernel, scores, iam) = go_up(x)

#         mask_features_inst = mask_features["mb_instance"]
#         mask_features_occl = mask_features["mb_occluder"]
        
#         inst_kernel = kernel["instance_kernel"]
#         # inst_bound_kernel = kernel["instance_bound_kernel"]
#         occl_kernel = kernel["occluder_kernel"]
#         # occl_bound_kernel = kernel["occluder_bound_kernel"]

#         # Predicting instance masks
#         N = inst_kernel.shape[1]  # num_masks
#         B, C, H, W = mask_features_inst.shape

#         masks = torch.bmm(
#             inst_kernel,    # (B, N, 128)
#             mask_features_inst.view(B, C, H * W)   # (B, 128, [HW])
#         ) # -> (B, N, [HW])
#         masks = masks.view(B, N, H, W)  # (B, N, [HW]) -> (B, N, H, W)

#         # masks_bounds = torch.bmm(
#         #     inst_bound_kernel,    # (B, N, 128)
#         #     mask_features.view(B, C, H * W)   # (B, 128, [HW])
#         # ) # -> (B, N, [HW])
#         # masks_bounds = masks_bounds.view(B, N, H, W)  # (B, N, [HW]) -> (B, N, H, W)

#         occluders = torch.bmm(
#             occl_kernel,    # (B, N, 128)
#             mask_features_occl.view(B, C, H * W)   # (B, 128, [HW])
#         ) # -> (B, N, [HW])
#         occluders = occluders.view(B, N, H, W)  # (B, N, [HW]) -> (B, N, H, W)

#         # occluders_bounds = torch.bmm(
#         #     occl_bound_kernel,    # (B, N, 128)
#         #     mask_features.view(B, C, H * W)   # (B, 128, [HW])
#         # ) # -> (B, N, [HW])
#         # occluders_bounds = occluders_bounds.view(B, N, H, W)  # (B, N, [HW]) -> (B, N, H, W)

#         inst_logits = logits["instance_logits"]
#         occl_logits = logits["occluder_logits"]

#         inst_scores = scores["instance_scores"]
#         occl_scores = scores["occluder_scores"]

#         # TODO: rename parameters
#         # TODO: pass in this format {"pred_masks": {"inst": ..., "occluder": ...}, "logits": {...}}
#         output = {
#             'pred_logits': inst_logits,
#             'pred_scores': inst_scores,
#             'pred_iam': iam,
#             'pred_masks': masks,  # instnace masks
#             'pred_kernel': kernel,
#             'pred_occluders': occluders, # occluders masks
#             'pred_logits_occluders': occl_logits,
#             'pred_scores_occluders': occl_scores,
#         }

        
#         # output = {
#         #     'pred_logits': logits,
#         #     'pred_scores': scores,
#         #     'pred_iam': iam,
#         #     'pred_masks': masks,  # instnace masks
#         #     'pred_kernel': kernel,
#         #     'pred_occluders': occluders, # occluders masks
#         #     # 'pred_occluders_bounds': occluders_bounds, # occluders bounds
#         #     # 'pred_masks_bounds': masks_bounds, # instnace bounds
#         # }
    
#         return output
