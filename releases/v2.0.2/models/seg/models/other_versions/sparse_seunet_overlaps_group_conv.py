import torch
from torch import nn

from ..heads.instance_head import InstanceBranch, PriorInstanceBranch, GroupInstanceBranch
from ..heads.mask_head import MaskBranch

from models.seg.modules.mixup import MixUpScaler
from models.seg.models.base import SparseSEUnet as BaseModel
from models.seg.models.base import DoubleConv, SE_block

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

        # overlap.
        # self.up_se_blocks_overlap = nn.ModuleList([])
        # self.pp_se_blocks_overlap = nn.ModuleList([])

        # self.up_conv_layers_overlap = nn.ModuleList([])
        # for _ in range(self.n_levels):
        #     # up convolution
        #     if len(self.up_conv_layers_overlap) == 0:
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

        #     self.up_conv_layers_overlap.append(upconv)

        #      # SE blocks following the upconv 
        #     up_se = SE_block(num_features=self.n_filters)            
        #     self.up_se_blocks_overlap.append(up_se)


        # self.overlap_conv = nn.Conv2d(
        #     208, #256, #(self.n_filters // 4) * 5 + 2 * self.n_filters,
        #     1,
        #     kernel_size=1,
        #     stride=1,
        # )
                
        
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


        self.prior_overlap_branch = PriorInstanceBranch(in_channels=208)

        # instance branch.
        # self.instance_branch = InstanceBranch(kernel_dim=self.kernel_dim, num_masks=self.num_masks)
        self.ovlp_branch = GroupInstanceBranch(
            dim=256, 
            kernel_dim=self.kernel_dim, 
            num_masks=self.num_masks
            )
        self.inst_branch = GroupInstanceBranch(
            dim=256 + self.num_masks, 
            kernel_dim=self.kernel_dim, 
            um_masks=self.num_masks
            )
        

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
        def go_up(x, overlap_feats):
            for i in range(self.n_levels):
                # if self.coord_conv:
                coord_features = self.compute_coordinates(x)
                x = torch.cat([coord_features, x], dim=1)
                # overlap_feats = torch.cat([coord_features, overlap_feats], dim=1)
                
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
                x = self.up_conv_layers[i](x)
                x = self.up_se_blocks[i](x)

                # overlap_feats = nn.UpsamplingBilinear2d(scale_factor=2)(overlap_feats)
                # overlap_feats = self.up_conv_layers_overlap[i](overlap_feats)
                # overlap_feats = self.up_se_blocks_overlap[i](overlap_feats)
                
                if self.pyramid_pooling:
                    x = torch.cat([x, down_pp_out_tensors[-(i + 1)]], dim=1)
                    # overlap_feats = torch.cat([overlap_feats, down_pp_out_tensors[-(i + 1)]], dim=1)
                else:
                    x = torch.cat([x, down_conv_out_tensors[-(i + 1)]], dim=1)
                    # overlap_feats = torch.cat([overlap_feats, down_conv_out_tensors[-(i + 1)]], dim=1)
                
                # x = x + overlap_feats
                
                # multi-level
                # if self.multi_level:
                if i != 0:
                    mb = nn.UpsamplingBilinear2d(scale_factor=2)(mb)    # (1, 128, 128, 128)
                    mb = torch.cat([x, mb], dim=1)
                    mb = self.mask_branch[i](mb)     
                else:
                    mb = self.mask_branch[i](x)

                if i != 0:
                    # scale: (B, N, Hx, Wx) -> (B, N, Hx * 2, Wx * 2)
                    # x features shape: (B, Di, Hx * 2, Wx * 2)
                    inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
                    inst_feats = torch.cat([x, inst_feats], dim=1)
                    inst_feats = self.prior_instance_branch[i](inst_feats)
                else:
                    # inst_feats shape: (B, Dm, Hx, Wx)
                    inst_feats = self.prior_instance_branch[i](x)

                    
                # single-level
                # else:
                # if i == self.n_levels - 1:
                #     mb = self.mask_branch[0](x)
                #     inst_feats = self.prior_instance_branch[0](x)
                        
                if i == self.n_levels - 1:
                    # _overlap_feats = torch.cat([overlap_feats, inst_feats], dim=1)
                    # _overlap_feats = self.prior_overlap_branch(overlap_feats)

                    # instance overlap feats
                    # _overlap_feats = torch.matmul(inst_feats, _overlap_feats)
                    # _overlap_feats = _overlap_feats * inst_feats
                    # _overlap_feats = _overlap_feats + inst_feats

                    ovlp_feats = inst_feats.clone()

                    logits, kernel, scores, ovlp_iam = self.ovlp_branch(ovlp_feats)
                    inst_feats = torch.cat([inst_feats, ovlp_iam], dim=1)

                    logits, kernel, scores, inst_iam = self.inst_branch(inst_feats, idx)

                    iam = {
                        "overlap_iam": ovlp_iam,
                        "isntnace_iam": inst_iam
                    }
                    # logits, kernel, scores, iam = self.instance_branch(inst_feats, _overlap_feats, idx)

            return x, overlap_feats, mb, (logits, kernel, scores, iam)
    
        # cyto
        overlap_feats = x.clone()
        x, overlap_feats, mask_features, (logits, kernel, scores, iam) = go_up(x, overlap_feats)
        
        # Predicting instance masks
        N = kernel.shape[1]  # num_masks
        B, C, H, W = mask_features.shape

        masks = torch.bmm(
            kernel,    # (B, N, 128)
            mask_features.view(B, C, H * W)   # (B, 128, [HW])
        ) # -> (B, N, [HW])
        masks = masks.view(B, N, H, W)  # (B, N, [HW]) -> (B, N, H, W)

        overlaps = self.overlap_conv(overlap_feats)  # (B, 1, H, W)
        
        output = {
            'pred_logits': logits,
            'pred_scores': scores,
            'pred_iam': iam,
            'pred_masks': masks,
            'pred_kernel': kernel,
            'pred_overlaps': overlaps
        }
    
        return output
