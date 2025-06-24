import torch
from torch import nn
from torch.nn import functional as F

import sys
sys.path.append("./")

from models.seg.decoders.iadecoder_v2.iadecoder import IADecoder
from configs.structure import Decoder
from utils.registry import HEADS, DECODERS

from omegaconf import OmegaConf


@DECODERS.register(name='iadecoder_ml_v2.1')
class IADecoder(IADecoder):
    def __init__(self, cfg: Decoder, embed_dims: list = [], n_levels: int = 4):
        super().__init__(cfg, embed_dims, n_levels)


        # instance branch.
        # self.instance_branch = nn.ModuleList([])
        # instance_branch_layer = HEADS.get(cfg.instance_branch.type)
        # for i in range(self.n_levels):
        #     instance_branch = instance_branch_layer(
        #         in_channels=embed_dims[i] + 2, 
        #         out_channels=self.inst_dim, 
        #         num_convs=self.num_convs
        #     )
        #     self.instance_branch.append(instance_branch)
        
        # instance head.
        self.instance_head = nn.ModuleList([])
        for i in range(self.n_levels):
            # cfg.instance_head.in_channels = embed_dims[i]

            # instance_head = OmegaConf.to_container(cfg.instance_head, resolve=True)
            # instance_head['in_res'] = (16 * (2 ** i), 16 * (2 ** i))

            instance_head = HEADS.build(cfg.instance_head)
            self.instance_head.append(instance_head)

        self._init_weights()
    

    def _forward(self, skips):
        aux_outputs = []

        for i in range(self.n_levels):
            if i != 0:
                coord_features = self.compute_coordinates(x)
                x = torch.cat([coord_features, x], dim=1)
                
                skip = skips[-(i + 1)]
                skip = self.skip_conv_layers[i](skip)

                x = nn.UpsamplingBilinear2d(scale_factor=2)(x)
                x = torch.cat([x, skip], dim=1)
                x = self.up_conv_layers[i](x)
            else:
                skip = skips[-1]
                skip = self.skip_conv_layers[i](skip)

                coord_features = self.compute_coordinates(skip)
                x = torch.cat([coord_features, skip], dim=1)
                x = self.up_conv_layers[i](x)


            if i != 0:
                results = self.instance_head[i](x, inst_embed)
                inst_embed = results["inst_feats"]['instance_feats']
            else:
                results = self.instance_head[i](x)
                inst_embed = results["inst_feats"]['instance_feats']

            x = results['pixel_feats']


            aux_output = self._set_aux_loss(results)
            aux_outputs.append(aux_output)



        mask_feats = self.projection(x)
        results["mask_feats"] = mask_feats
        results["inst_feats"] = x

        results["aux_outputs"] = aux_outputs[:-1]
    
        return results


    
    def _set_aux_loss(self, results):
        logits = results["logits"]
        scores = results["objectness_scores"]
        inst_kernel = results["kernels"]["instance_kernel"]
        bboxes = results["bboxes"]['instance_bboxes'].sigmoid()
        # mask_feats = results["mask_feats"]
        # inst_feats = results["inst_feats"]
        inst_masks = results["masks"]["instance_masks"]

        output = {
            'pred_logits': logits,
            'pred_scores': scores,
            'pred_iams': results['iams'],
            'pred_instance_masks': inst_masks,
            'pred_bboxes': bboxes,
        }
    
        return output
    
    
    def process_outputs(self, results, ori_shape):
        logits = results["logits"]
        scores = results["objectness_scores"]
        inst_kernel = results["kernels"]["instance_kernel"]
        bboxes = results["bboxes"]['instance_bboxes'].sigmoid()
        mask_feats = results["mask_feats"]
        inst_feats = results["inst_feats"]
        inst_masks = results["masks"]["instance_masks"]
        
        inst_masks = F.interpolate(inst_masks, size=ori_shape, 
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
            },

            'aux_outputs': results["aux_outputs"]
        }
    
        return output
    
    