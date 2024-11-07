import sys
import hydra

sys.path.append("./")

from configs import cfg
from models.build_model import get_model
from thop import profile
import torch


def get_flops(model, device="cuda:0"):
    # get-flops
    model.eval()
    
    x = torch.randn(1, 3, 512, 512)
    if torch.cuda.is_available():
        x = x.to(device)
    flops, params = profile(model, inputs=(x,), verbose=True)

    gflops = flops / 1e9
    params_in_millions = params / 1e6
    print(f'Flops: {gflops:.3f}G')
    print(f'Params: {params_in_millions:.3f}M')
    print()


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def profile_model(cfg: cfg):

    if True:
        # inst head.
        # cfg.model.decoder.instance_head.type = "InstanceHead-v1.1"
        # cfg.model.decoder.instance_head.type = "InstanceHead-v2.1-attn"
        # cfg.model.decoder.instance_head.type = "InstanceHead-v2.2-two-way-attn"
        # cfg.model.decoder.instance_head.type = "InstanceHead-v2.2.1-dual-update"
        # cfg.model.decoder.instance_head.type = "InstanceHead-v2.2.3-dual-update"
        cfg.model.decoder.instance_head.type = "InstanceHead-v2.2.a-removed-inst-feats"
        # cfg.model.decoder.instance_head.type = "InstanceHead-v3.0-multi-query"
        cfg.model.decoder.instance_head.in_channels = 256
        cfg.model.decoder.instance_head.kernel_dim = 256
        cfg.model.decoder.instance_head.num_groups = 1
        cfg.model.decoder.instance_head.num_masks = 100
        # cfg.model.decoder.instance_head.num_layers = 1
        # mask branch.
        cfg.model.decoder.mask_branch.type = "MaskStackedConv" # MaskDoubleConv, MaskStackedConv
        cfg.model.decoder.mask_branch.dim = 256
        # inst branch.
        cfg.model.decoder.instance_branch.type = "InstStackedConv" # InstDoubleConv, InstStackedConv
        cfg.model.decoder.instance_branch.dim = 256
        
        # model.
        # cfg.model.type = "resnet_iaunet_multitask_ml"
        cfg.model.type = "iaunet"
        cfg.model.decoder.type = "iadecoder_ml_fpn"
        # cfg.model.encoder = dict(
        #     type='ResNet',
        #     depth=50,
        #     num_stages=4,
        #     out_indices=[1, 2, 3, 4],
        #     pretrained=True,
        # )

        cfg.model.n_levels = 4
        cfg.model.decoder.num_convs = 2
        cfg.model.decoder.last_layer_only = False

    model = get_model(cfg)

    if torch.cuda.is_available():
        model = model.to("cuda:0")

    get_flops(model)


if __name__ == "__main__":
    profile_model()



        # cfg.model.encoder = dict(
        #     type='UNet',
        #     num_stages=5,
        #     out_indices=(0, 1, 2, 3, 4),
        #     pyramid_pooling=False,
        #     pp_embed_dim=128,
        #     embed_dims=[64, 128, 256, 512, 1024],
        #     # embed_dims=[32, 64, 128, 256, 512],
        #     depths=[1, 1, 1, 2, 1]
        # )