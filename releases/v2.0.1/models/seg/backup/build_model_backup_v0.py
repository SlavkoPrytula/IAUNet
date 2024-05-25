import os
import torch

from models.seg.models.sparse_seunet import SparseSEUnet
from models.seg.models.sparse_seunet_add_overlaps import SparseSEUnet as SparseSEUnetAddOverlaps
from models.seg.models.sparse_seunet_cat_overlaps import SparseSEUnet as SparseSEUnetCatOverlaps
from models.seg.models.sparse_seunet_kernel_fusion import SparseSEUnet as SparseSEUnetKernelFusion
from models.seg.models.sparse_seunet_feat_iam_mix import SparseSEUnet as SparseSEUnetFeatIAMMix

from configs import cfg


def build_model(cfg: cfg):
    models = {
        'sparse_seunet': SparseSEUnet,
        'sparse_seunet_kernel_fusion': SparseSEUnetKernelFusion,
        'sparse_seunet_feat_iam_mix': SparseSEUnetFeatIAMMix,
        'sparse_seunet_add_overlaps': SparseSEUnetAddOverlaps,
        'sparse_seunet_cat_overlaps': SparseSEUnetCatOverlaps
    }

    model = models[cfg.model.arch](
        cfg=cfg,
        )
    
    if cfg.model.load_pretrained:
        print(f"- Loading pretrained weights:\n[{cfg.model.weights}]")

        current_model_dict = model.state_dict()
        loaded_state_dict = torch.load(cfg.model.weights)

        # debugging:
        # for k, v in loaded_state_dict.items():
        #     if k in current_model_dict and v.size() == current_model_dict[k].size():
        #         print(f'loading from checkpoint:\n- {k}')
        #     else:
        #         print(f'couldnt load:\n- {k}')


        # new_state_dict={k: v if v.size() == current_model_dict[k].size() else current_model_dict[k] 
        #                 for k, v in zip(current_model_dict.keys(), loaded_state_dict.values())}

        for k, v in loaded_state_dict.items():
            if k in current_model_dict and v.size() == current_model_dict[k].size():
                current_model_dict[k] = v
            elif k in current_model_dict:
                print(f"WARNING: Skipping loading weights for parameter '{k}' due to size mismatch.")
                print(f"Expected size: {current_model_dict[k].size()}, but got size: {v.size()}")
            else:
                print(f"WARNING: Skipping loading weights for parameter '{k}' as it was not found in the current model.")

        
        # Warn about weights in the current model but not present in the pretrained weights
        for k in current_model_dict.keys():
            if k not in loaded_state_dict:
                print(f"WARNING: Parameter '{k}' in the current model is not present in the pretrained weights.")


        model.load_state_dict(current_model_dict, strict=False)

        # model.load_state_dict(torch.load(cfg.model.weights), strict=False)
        print("- Weights loaded!")
        
    model.to(cfg.device)

    return model


def load_model(cfg: cfg, path: str):
    model = build_model(cfg)
    model.load_state_dict(torch.load(path), strict=True)
    
    model.to(cfg.device)
    model.eval()
    print('- weights loaded!')

    return model


if __name__ == '__main__':
    from configs.base import cfg
    model = build_model(cfg)

    x = torch.randn(1, 1, 128, 128).to(cfg.device)
    out = model(x)
    print(out.shape)
