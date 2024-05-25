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


        # overlaps.
        # self.up_se_blocks_overlaps = nn.ModuleList([])
        # self.up_conv_layers_overlaps = nn.ModuleList([])
        # for _ in range(self.n_levels):
        #     # up convolution
        #     if len(self.up_conv_layers_overlaps) == 0:
        #         # if self.coord_conv:
        #         upconv = DoubleConv(self.n_filters+2, self.n_filters)
        #         # else:
        #         #     upconv = DoubleConv(self.n_filters, self.n_filters)
        #     else:
        #         # if self.coord_conv:
        #         upconv = DoubleConv(
        #             (self.n_filters // 4) * 5 + 2 * self.n_filters+2, self.n_filters
        #         )
        #         # else:
        #         #     upconv = DoubleConv(
        #         #         (self.n_filters // 4) * 5 + 2 * self.n_filters, self.n_filters
        #         #     )
        #     self.up_conv_layers_overlaps.append(upconv)

        #      # SE blocks following the upconv 
        #     up_se = SE_block(num_features=self.n_filters)            
        #     self.up_se_blocks_overlaps.append(up_se)


        # mask branch.
        self.mask_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                self.mask_branch.append(MaskBranch(208))
            else:
                self.mask_branch.append(MaskBranch(208+128))
        
        # instance features.
        self.prior_instance_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                self.prior_instance_branch.append(PriorInstanceBranch(in_channels=208))
            else:
                self.prior_instance_branch.append(PriorInstanceBranch(in_channels=464))


        # occluder.
        # mask branch.
        self.mask_branch_occluder = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                self.mask_branch_occluder.append(MaskBranch(208))
            else:
                self.mask_branch_occluder.append(MaskBranch(208+128))
        
        # instance features.
        self.prior_occluder_branch = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                self.prior_occluder_branch.append(PriorInstanceBranch(in_channels=208))
            else:
                self.prior_occluder_branch.append(PriorInstanceBranch(in_channels=464))


        # overlap.
        # mask branch.
        self.mask_branch_overlap = nn.ModuleList([])
        for i in range(self.n_levels):
            if i == 0:
                self.mask_branch_overlap.append(MaskBranch(208))
            else:
                self.mask_branch_overlap.append(MaskBranch(208+128))
        
        # instance features.
        # self.prior_overlap_branch = nn.ModuleList([])
        # self.prior_overlap_branch.append(PriorInstanceBranch(in_channels=464))
        # for i in range(self.n_levels):
        #     if i == 0:
        #         self.prior_overlap_branch.append(PriorInstanceBranch(in_channels=208))
        #     else:
        #         self.prior_overlap_branch.append(PriorInstanceBranch(in_channels=464))

        # self.occluder_context_block = ContextBlock(inplanes=464, ratio=4, pooling_type='att', fusion_types=['channel_add', 'channel_mul'])
        # self.instance_context_block = ContextBlock(inplanes=464, ratio=4, pooling_type='att', fusion_types=['channel_add', 'channel_mul'])
        # self.out_occluder_branch = PriorInstanceBranch(in_channels=464, out_channels=256)
        # self.out_instance_branch = PriorInstanceBranch(in_channels=464, out_channels=256)

        # instance branch.
        # self.instance_branch = InstanceBranch(kernel_dim=self.kernel_dim, num_masks=self.num_masks)
        self.occluder_branch = GroupInstanceBranch(
            dim=256, 
            kernel_dim=self.kernel_dim, 
            num_masks=self.num_masks
            )
        # self.occluder_bound_branch = GroupInstanceBranch(
        #     dim=256, 
        #     kernel_dim=self.kernel_dim, 
        #     num_masks=self.num_masks
        #     )

        self.overlap_branch = GroupInstanceBranch(
            dim=256, 
            kernel_dim=self.kernel_dim, 
            num_masks=self.num_masks
            )
        
        self.instance_branch = GroupInstanceBranch(
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
            # x_ovlp = x.clone()

            for i in range(self.n_levels):
                # if self.coord_conv:
                coord_features = self.compute_coordinates(x)
                x = torch.cat([coord_features, x], dim=1)
                x_occl = torch.cat([coord_features, x_occl], dim=1)
                # x_ovlp = torch.cat([coord_features, x_ovlp], dim=1)
                
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
                x = self.up_conv_layers[i](x)
                x = self.up_se_blocks[i](x)

                x_occl = nn.UpsamplingBilinear2d(scale_factor=2)(x_occl)
                x_occl = self.up_conv_layers_occluders[i](x_occl)
                x_occl = self.up_se_blocks_occluders[i](x_occl)

                # x_ovlp = nn.UpsamplingBilinear2d(scale_factor=2)(x_ovlp)
                # x_ovlp = self.up_conv_layers_overlaps[i](x_ovlp)
                # x_ovlp = self.up_se_blocks_overlaps[i](x_ovlp)


                if self.pyramid_pooling:
                    x = torch.cat([x, down_pp_out_tensors[-(i + 1)]], dim=1)
                    x_occl = torch.cat([x_occl, down_pp_out_tensors[-(i + 1)]], dim=1)
                    # x_ovlp = torch.cat([x_ovlp, down_pp_out_tensors[-(i + 1)]], dim=1)
                else:
                    x = torch.cat([x, down_conv_out_tensors[-(i + 1)]], dim=1)
                    x_occl = torch.cat([x_occl, down_conv_out_tensors[-(i + 1)]], dim=1)
                    # x_ovlp = torch.cat([x_ovlp, down_conv_out_tensors[-(i + 1)]], dim=1)
                
                # multi-level
                # if self.multi_level:
                if i != 0:
                    mb_inst = nn.UpsamplingBilinear2d(scale_factor=2)(mb_inst)    # (1, 128, 128, 128)
                    mb_inst = torch.cat([x, mb_inst], dim=1)
                    mb_inst = self.mask_branch[i](mb_inst)     

                    mb_occl = nn.UpsamplingBilinear2d(scale_factor=2)(mb_occl)    # (1, 128, 128, 128)
                    mb_occl = torch.cat([x_occl, mb_occl], dim=1)
                    mb_occl = self.mask_branch_occluder[i](mb_occl)    

                    # mb_ovlp = nn.UpsamplingBilinear2d(scale_factor=2)(mb_ovlp)    # (1, 128, 128, 128)
                    # mb_ovlp = torch.cat([x_ovlp, mb_ovlp], dim=1)
                    # mb_ovlp = self.mask_branch_overlap[i](mb_ovlp)    
                else:
                    mb_inst = self.mask_branch[i](x)
                    mb_occl = self.mask_branch_occluder[i](x_occl)
                    # mb_ovlp = self.mask_branch_overlap[i](x_ovlp)

                if i != 0:
                    # scale: (B, N, Hx, Wx) -> (B, N, Hx * 2, Wx * 2)
                    # x features shape: (B, Di, Hx * 2, Wx * 2)
                    inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
                    inst_feats = torch.cat([x, inst_feats], dim=1)

                    occl_feats = nn.UpsamplingBilinear2d(scale_factor=2)(occl_feats)
                    occl_feats = torch.cat([x_occl, occl_feats], dim=1)

                    # ovlp_feats = nn.UpsamplingBilinear2d(scale_factor=2)(ovlp_feats)
                    # ovlp_feats = torch.cat([x_ovlp, ovlp_feats], dim=1)

                    if i != self.n_levels - 1:
                        inst_feats = self.prior_instance_branch[i](inst_feats)
                        occl_feats = self.prior_occluder_branch[i](occl_feats)
                        # ovlp_feats = self.prior_overlap_branch[i](ovlp_feats)
                else:
                    # inst_feats shape: (B, Dm, Hx, Wx)
                    inst_feats = self.prior_instance_branch[i](x)
                    occl_feats = self.prior_occluder_branch[i](x_occl)
                    # ovlp_feats = self.prior_overlap_branch[i](x_ovlp)

                    
                # single-level
                # else:
                # if i == self.n_levels - 1:
                #     mb = self.mask_branch[0](x)
                #     inst_feats = self.prior_instance_branch[0](x)
                        
                if i == self.n_levels - 1:
                    # occluder.
                    # occl_feats = inst_feats.clone()
                    # occl_feats = self.occluder_context_block(occl_feats)
                    occl_feats = self.prior_occluder_branch[i](occl_feats)  # (256, H, W)

                    # logits, occl_bound_kernel, scores, occl_bound_iam = self.occluder_bound_branch(occl_feats)
                    occl_logits, occl_kernel, occl_scores, occl_iam = self.occluder_branch(occl_feats)

                    # overlap.
                    # ovlp_feats = inst_feats + occl_feats
                    # ovlp_feats = self.prior_overlap_branch[0](ovlp_feats)  # (256, H, W)
                    # ovlp_logits, ovlp_kernel, ovlp_scores, ovlp_iam = self.overlap_branch(ovlp_feats)

                    # instnace.
                    # inst_feats = inst_feats + occl_feats
                    # inst_feats
                    # inst_feats = self.instance_context_block(inst_feats)
                    inst_feats = self.prior_instance_branch[i](inst_feats) # (256, H, W)


                    ovlp_feats = inst_feats + occl_feats
                    ovlp_logits, ovlp_kernel, ovlp_scores, ovlp_iam = self.overlap_branch(ovlp_feats)


                    inst_feats = inst_feats + occl_feats # (256, H, W)
                    
                    # logits, inst_bound_kernel, scores, inst_bound_iam = self.instance_bound_branch(inst_feats)
                    inst_logits, inst_kernel, inst_scores, inst_iam = self.instance_branch(inst_feats)


                    
                    mb = {
                        "mb_occluder": mb_occl,
                        "mb_overlap": mb_inst + mb_occl, # mb_ovlp,
                        "mb_instance": mb_inst
                    }
                    
                    logits = {
                        "occluder_logits": occl_logits,
                        "overlap_logits": ovlp_logits,
                        "instance_logits": inst_logits,
                    }
                    scores = {
                        "occluder_scores": occl_scores,
                        "overlap_scores": ovlp_scores,
                        "instance_scores": inst_scores,
                    }

                    kernel = {
                        "occluder_kernel": occl_kernel,
                        "overlap_kernel": ovlp_kernel,
                        # "occluder_bound_kernel": occl_bound_kernel,
                        "instance_kernel": inst_kernel,
                        # "instance_bound_kernel": inst_bound_kernel
                    }

                    iam = {
                        "occluder_iam": occl_iam,
                        "overlap_iam": ovlp_iam,
                        # "occluder_bound_iam": occl_bound_iam,
                        "instance_iam": inst_iam,
                        # "instance_bound_iam": inst_bound_iam
                    }

            return x, mb, (logits, kernel, scores, iam)
    
        # cyto
        x, mask_features, (logits, kernel, scores, iam) = go_up(x)

        mask_features_inst = mask_features["mb_instance"]
        mask_features_occl = mask_features["mb_occluder"]
        mask_features_ovlp = mask_features["mb_overlap"]
        
        inst_kernel = kernel["instance_kernel"]
        # inst_bound_kernel = kernel["instance_bound_kernel"]
        occl_kernel = kernel["occluder_kernel"]
        # occl_bound_kernel = kernel["occluder_bound_kernel"]
        ovlp_kernel = kernel["overlap_kernel"]

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

        overlaps = torch.bmm(
            ovlp_kernel,    # (B, N, 128)
            mask_features_ovlp.view(B, C, H * W)   # (B, 128, [HW])
        ) # -> (B, N, [HW])
        overlaps = overlaps.view(B, N, H, W)  # (B, N, [HW]) -> (B, N, H, W)


        inst_logits = logits["instance_logits"]
        occl_logits = logits["occluder_logits"]
        ovlp_logits = logits["overlap_logits"]

        inst_scores = scores["instance_scores"]
        occl_scores = scores["occluder_scores"]
        ovlp_scores = scores["overlap_scores"]

        # TODO: rename parameters
        # TODO: pass in this format {"pred_masks": {"inst": ..., "occluder": ...}, "logits": {...}}
        output = {
            'pred_logits': inst_logits,
            'pred_scores': inst_scores,
            'pred_iam': iam,
            'pred_masks': masks,  # instnace masks
            'pred_kernel': kernel,

            'pred_occluders': occluders, # occluders masks
            'pred_logits_occluders': occl_logits,
            'pred_scores_occluders': occl_scores,

            'pred_overlaps': overlaps, # occluders masks
            'pred_logits_overlaps': ovlp_logits,
            'pred_scores_overlaps': ovlp_scores,
        }

        
        # output = {
        #     'pred_logits': logits,
        #     'pred_scores': scores,
        #     'pred_iam': iam,
        #     'pred_masks': masks,  # instnace masks
        #     'pred_kernel': kernel,
        #     'pred_occluders': occluders, # occluders masks
        #     # 'pred_occluders_bounds': occluders_bounds, # occluders bounds
        #     # 'pred_masks_bounds': masks_bounds, # instnace bounds
        # }
    
        return output
