import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn.weight_init import c2_msra_fill

import sys
sys.path.append("./")

from models.seg.heads.instance_head import InstanceBranch, PriorInstanceBranch, GroupInstanceBranch
from models.seg.heads.mask_head import MaskBranch

# from models.seg.modules.mixup import MixUpScaler
from models.seg.models.base import SparseSEUnet as BaseModel
from models.seg.models.base import DoubleConv, SE_block
# from models.seg.blocks import ContextBlock
# from models.seg.blocks import GCN, Conv2d
# from models.seg.layers import get_norm

import fvcore.nn.weight_init as weight_init

from configs import cfg
from utils.registry import MODELS


@MODELS.register(name="sparse_seunet_occluder_gcn")
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
        self.up_conv_layers = nn.ModuleList([])
        # CHANGED: removed additional +2 channels
        # CHANGED: moved the coordinate features addition to instance branches
        for _ in range(self.n_levels):
            # up convolutions
            if len(self.up_conv_layers) == 0:
                upconv = DoubleConv(self.n_filters, self.n_filters)
            else:
                upconv = DoubleConv(
                    (self.n_filters // 4) * 5 + 2 * self.n_filters, self.n_filters
                )
            self.up_conv_layers.append(upconv)

        
        # mask branch.
        self.mask_branch = nn.ModuleList([])
        self.mask_branch.append(MaskBranch(in_channels=208 + 2, out_channels=256, num_convs=4))
        # for i in range(self.n_levels):
        #     if i == 0:
        #         self.mask_branch.append(MaskBranch(in_channels=208 + 2, num_convs=self.num_convs))
        #     else:
        #         self.mask_branch.append(MaskBranch(in_channels=464 + 2, num_convs=self.num_convs))
        
        # mask feature map projection layer (1x1)
        self.projection = nn.Conv2d(256, self.kernel_dim, kernel_size=1)


        # instance features.
        self.prior_instance_branch = nn.ModuleList([])
        self.prior_instance_branch.append(PriorInstanceBranch(in_channels=208 + 2, out_channels=256, num_convs=4))
        # for i in range(self.n_levels):
        #     if i == 0:
        #         self.prior_instance_branch.append(PriorInstanceBranch(in_channels=208 + 2, out_channels=256, num_convs=self.num_convs))
        #     else:
        #         self.prior_instance_branch.append(PriorInstanceBranch(in_channels=464 + 2, out_channels=256, num_convs=self.num_convs))


        self.prior_occluder_branch = nn.ModuleList([])
        self.prior_occluder_branch.append(PriorInstanceBranch(in_channels=208 + 2, out_channels=256, num_convs=4))


        # instance branch.
        self.occluder_branch = InstanceBranch(dim=256 + 2, kernel_dim=self.kernel_dim, num_masks=self.num_masks)
        self.instance_branch = InstanceBranch(dim=256 + 2, kernel_dim=self.kernel_dim, num_masks=self.num_masks)


        self.init_weights()

    def init_weights(self):
        c2_msra_fill(self.projection)

        

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
            for i in range(self.n_levels):
                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
                x = self.up_conv_layers[i](x)
                x = self.up_se_blocks[i](x)

                # x_occl = nn.UpsamplingBilinear2d(scale_factor=2)(x_occl)
                # x_occl = self.up_conv_layers_occluders[i](x_occl)
                # x_occl = self.up_se_blocks_occluders[i](x_occl)

                if self.pyramid_pooling:
                    x = torch.cat([x, down_pp_out_tensors[-(i + 1)]], dim=1)
                    # x_occl = torch.cat([x_occl, down_pp_out_tensors[-(i + 1)]], dim=1)
                else:
                    x = torch.cat([x, down_conv_out_tensors[-(i + 1)]], dim=1)
                    # x_occl = torch.cat([x_occl, down_conv_out_tensors[-(i + 1)]], dim=1)
                
                # multi-level
                # if self.multi_level:
                # if i != 0:
                #     mb_inst = nn.UpsamplingBilinear2d(scale_factor=2)(mb_inst)    # (1, 128, 128, 128)
                #     mb_inst = torch.cat([x, mb_inst], dim=1)
                #     mb_inst = self.mask_branch[i](mb_inst)     

                #     # mb_occl = nn.UpsamplingBilinear2d(scale_factor=2)(mb_occl)    # (1, 128, 128, 128)
                #     # mb_occl = torch.cat([x_occl, mb_occl], dim=1)
                #     # mb_occl = self.mask_branch_occluder[i](mb_occl)    
                # else:
                #     mb_inst = self.mask_branch[i](x)
                #     # mb_occl = self.mask_branch_occluder[i](x_occl)

                # if i != 0:
                #     # scale: (B, N, Hx, Wx) -> (B, N, Hx * 2, Wx * 2)
                #     # x features shape: (B, Di, Hx * 2, Wx * 2)
                #     inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
                #     inst_feats = torch.cat([x, inst_feats], dim=1)

                #     # occl_feats = nn.UpsamplingBilinear2d(scale_factor=2)(occl_feats)
                #     # occl_feats = torch.cat([x_occl, occl_feats], dim=1)

                #     if i != self.n_levels - 1:
                #         inst_feats = self.prior_instance_branch[i](inst_feats)
                #         # occl_feats = self.prior_occluder_branch[i](occl_feats)
                # else:
                #     # inst_feats shape: (B, Dm, Hx, Wx)
                #     inst_feats = self.prior_instance_branch[i](x)
                #     # occl_feats = self.prior_occluder_branch[i](x_occl)

                    
                # # single-level
                # # else:
                # # if i == self.n_levels - 1:
                # #     mb = self.mask_branch[0](x)
                # #     inst_feats = self.prior_instance_branch[0](x)
                        
                # if i == self.n_levels - 1:
                #     # (conv) --> (conv + gcn) --> (conv) --> (conv)
                #     #     0                1          2          3

                #     # occl_feats = self.prior_occluder_branch[i](occl_feats)  # (256, H, W)
                #     inst_feats = self.prior_instance_branch[i](inst_feats)  # (256, H, W)

                #     occl_feats = inst_feats.clone()

                #     for cnt, layer in enumerate(self.conv_norm_relu_occluder):
                #         occl_feats = layer(occl_feats)  # conv layer

                #         if cnt == 1 and len(occl_feats) != 0:
                #             occl_feats = self.gcn_occluder(occl_feats)

                #     # occl_feats: (256, H, W)
                #     # inst_feats: (256, H, W)
                #     _occl_feats = occl_feats.clone()

                #     inst_feats = inst_feats + _occl_feats # (256, H, W)

                #     for cnt, layer in enumerate(self.conv_norm_relu_instance):
                #         inst_feats = layer(inst_feats)

                #         if cnt == 1 and len(inst_feats) != 0:
                #             inst_feats = self.gcn_instance(inst_feats)

                #     occl_logits, occl_kernel, occl_scores, occl_iam = self.occluder_branch(occl_feats)
                #     inst_logits, inst_kernel, inst_scores, inst_iam = self.instance_branch(inst_feats)

                
            coord_features = self.compute_coordinates(x)
            x = torch.cat([coord_features, x], dim=1)

            # mask feats.
            mb = self.mask_branch[0](x)
            mb = self.projection(mb)
            
            # instance feats. -- kernel
            occl_feats = self.prior_occluder_branch[0](x)
            inst_feats = self.prior_instance_branch[0](x)
            
            # occluder.
            coord_features = self.compute_coordinates(occl_feats)
            occl_feats = torch.cat([coord_features, occl_feats], dim=1)
            occl_logits, occl_kernel, occl_scores, occl_iam = self.occluder_branch(occl_feats)
            
            # instance.
            coord_features = self.compute_coordinates(inst_feats)
            inst_feats = torch.cat([coord_features, inst_feats], dim=1)
            inst_logits, inst_kernel, inst_scores, inst_iam = self.instance_branch(inst_feats)
            

            mb = {
                # "mb_occluder": mb_occl,
                "mb_instance": mb
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

            return x, mb, (logits, kernel, scores, iam)
    
        # cyto
        x, mask_features, (logits, kernel, scores, iam) = go_up(x)

        mask_features_inst = mask_features["mb_instance"]
        # mask_features_occl = mask_features["mb_occluder"]
        
        inst_kernel = kernel["instance_kernel"]
        # inst_bound_kernel = kernel["instance_bound_kernel"]
        occl_kernel = kernel["occluder_kernel"]
        # occl_bound_kernel = kernel["occluder_bound_kernel"]

        # Predicting instance masks
        _, N, D = inst_kernel.shape 
        B, C, H, W = mask_features_inst.shape

        # kernel: (1, N, D) -> (N, D, 1, 1)
        masks = []
        for b in range(len(inst_kernel)):
            m = mask_features_inst[b].unsqueeze(0)
            k = inst_kernel[b]
            k = k.view(N, D, 1, 1)

            inst = F.conv2d(m, k, stride=1)
            masks.append(inst)
        masks = torch.cat(masks, dim=0)


        # kernel: (1, N, D) -> (N, D, 1, 1)
        occluders = []
        for b in range(len(occl_kernel)):
            m = mask_features_inst[b].unsqueeze(0)
            k = occl_kernel[b]
            k = k.view(N, D, 1, 1)

            inst = F.conv2d(m, k, stride=1)
            occluders.append(inst)
        occluders = torch.cat(occluders, dim=0)

        # masks = torch.bmm(
        #     inst_kernel,    # (B, N, 128)
        #     mask_features_inst.view(B, C, H * W)   # (B, 128, [HW])
        # ) # -> (B, N, [HW])
        # masks = masks.view(B, N, H, W)  # (B, N, [HW]) -> (B, N, H, W)

        # occluders = torch.bmm(
        #     occl_kernel,    # (B, N, 128)
        #     mask_features_inst.view(B, C, H * W)   # (B, 128, [HW])
        # ) # -> (B, N, [HW])
        # occluders = occluders.view(B, N, H, W)  # (B, N, [HW]) -> (B, N, H, W)

        inst_logits = logits["instance_logits"]
        occl_logits = logits["occluder_logits"]

        inst_scores = scores["instance_scores"]
        occl_scores = scores["occluder_scores"]

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
        }
    
        return output


if __name__ == "__main__":
    model = SparseSEUnet(cfg)
    x = torch.rand(2, 2, 512, 512)
    out = model(x)
    print(out["pred_masks"].shape)
    print(out["pred_occluders"].shape)