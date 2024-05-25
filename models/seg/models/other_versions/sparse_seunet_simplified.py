import torch
from torch import nn

import sys
sys.path.append("./")

from models.seg.heads.instance_head import InstanceBranch, PriorInstanceBranch, GroupInstanceBranch
from models.seg.heads.mask_head import MaskBranch

# from ..heads.instance_head import InstanceBranch, PriorInstanceBranch, GroupInstanceBranch
# from ..heads.mask_head import MaskBranch

from models.seg.models.base import SparseSEUnet as BaseModel
from models.seg.models.base import DoubleConv, SE_block

from configs import cfg

from utils.registry import MODELS


@MODELS.register(name="sprase_seunet_simplified")
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


        # inst.
        self.up_se_blocks_inst = nn.ModuleList([])
        self.up_conv_layers_inst = nn.ModuleList([])
        for _ in range(self.n_levels):
            # up convolution
            if len(self.up_conv_layers_inst) == 0:
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
            self.up_conv_layers_inst.append(upconv)

             # SE blocks following the upconv 
            up_se = SE_block(num_features=self.n_filters)            
            self.up_se_blocks_inst.append(up_se)

        
        # # mask branch.
        self.mask_branch = nn.ModuleList([])
        self.mask_branch.append(MaskBranch(208))
        # for i in range(self.n_levels):
        #     if i == 0:
        #         self.mask_branch.append(MaskBranch(64, num_convs=self.num_convs))
        #     else:
        #         self.mask_branch.append(MaskBranch(64+128, num_convs=self.num_convs))
        
        # # instance features.
        self.prior_instance_branch = nn.ModuleList([])
        self.prior_instance_branch.append(PriorInstanceBranch(in_channels=208, out_channels=256, num_convs=4))
        # for i in range(self.n_levels):
        #     if i == 0:
        #         self.prior_instance_branch.append(PriorInstanceBranch(in_channels=64, out_channels=256, num_convs=self.num_convs))
        #     else:
        #         self.prior_instance_branch.append(PriorInstanceBranch(in_channels=64+256, out_channels=256, num_convs=self.num_convs))

        # self.out_conv_mask = nn.Conv2d(
        #     64+128,
        #     128,
        #     kernel_size=1,
        #     stride=1,
        # )

        # self.out_conv_inst = nn.Conv2d(
        #     64+256,
        #     256,
        #     kernel_size=1,
        #     stride=1,
        # )

        # instance branch.
        self.instance_branch = InstanceBranch(dim=256, kernel_dim=self.kernel_dim, num_masks=self.num_masks)


        for modules in [self.up_conv_layers_inst, self.up_se_blocks_inst]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)
        
        

    # TESTING: add instance and mask branches only to the final layer of the decoder
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
            mb = x.clone()
            inst_feats = x.clone()

            for i in range(self.n_levels):
                # if self.coord_conv:
                coord_features = self.compute_coordinates(mb)
                mb = torch.cat([coord_features, mb], dim=1)
                inst_feats = torch.cat([coord_features, inst_feats], dim=1)
                
                mb = nn.UpsamplingBilinear2d(scale_factor=2)(mb)
                mb = self.up_conv_layers[i](mb)
                mb = self.up_se_blocks[i](mb)
                
                inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
                inst_feats = self.up_conv_layers_inst[i](inst_feats)
                inst_feats = self.up_se_blocks_inst[i](inst_feats)
                
                if self.pyramid_pooling:
                    mb = torch.cat([mb, down_pp_out_tensors[-(i + 1)]], dim=1)
                    inst_feats = torch.cat([inst_feats, down_pp_out_tensors[-(i + 1)]], dim=1)
                else:
                    mb = torch.cat([mb, down_conv_out_tensors[-(i + 1)]], dim=1)
                    inst_feats = torch.cat([inst_feats, down_conv_out_tensors[-(i + 1)]], dim=1)
                
                
                # multi-level
                # if self.multi_level:
                # if i != 0:
                #     mb = nn.UpsamplingBilinear2d(scale_factor=2)(mb)    # (1, 128, 128, 128)
                #     # print(down_pp_out_tensors[-(i + 1)].shape)
                #     # print(mb.shape)
                #     # raise
                #     # mb = self.mask_branch[i](mb)     
                # else:
                #     mb = nn.UpsamplingBilinear2d(scale_factor=2)(x)    # (1, 128, 128, 128)
            

                # if i != 0:
                #     # scale: (B, N, Hx, Wx) -> (B, N, Hx * 2, Wx * 2)
                #     # x features shape: (B, Di, Hx * 2, Wx * 2)
                #     inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
                #     # inst_feats = self.prior_instance_branch[i](inst_feats)
                # else:
                #     # inst_feats shape: (B, Dm, Hx, Wx)
                #     inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(x)



                # mb = nn.UpsamplingBilinear2d(scale_factor=2)(mb)
                # mb = self.mask_branch[i](mb) 
                # mb = torch.cat([mb, down_conv_out_tensors[-(i + 1)]], dim=1) # skip-connection (B, 64, H, W)

                # inst_feats = nn.UpsamplingBilinear2d(scale_factor=2)(inst_feats)
                # inst_feats = self.prior_instance_branch[i](inst_feats)
                # inst_feats = torch.cat([inst_feats, down_conv_out_tensors[-(i + 1)]], dim=1) # skip-connection (B, 64, H, W)
                    
                # single-level
                # else:
                # if i == self.n_levels - 1:
                #     mb = self.mask_branch[0](x)
                #     inst_feats = self.prior_instance_branch[0](x)
                        
                if i == self.n_levels - 1:
                    mb = self.mask_branch[-1](mb) # -> (B, 128, H, W)
                    inst_feats = self.prior_instance_branch[-1](inst_feats) # -> (B, 256, H, W)

                    # mb = self.out_conv_mask(mb)
                    # inst_feats = self.out_conv_inst(inst_feats)

                    logits, kernel, scores, iam = self.instance_branch(inst_feats, idx)

            return x, mb, (logits, kernel, scores, iam)
    
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



if __name__ == "__main__":
    model = SparseSEUnet(cfg)
    x = torch.rand(1, 2, 512, 512)
    out = model(x)
    print(out["pred_masks"].shape)
