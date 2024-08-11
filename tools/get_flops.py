import sys
import hydra

sys.path.append("./")

from configs import cfg
from models.build_model import get_model
from thop import profile
import torch


def get_flops(model, device="cuda:0"):
    # get-flops
    x = torch.randn(1, 3, 512, 512)
    if torch.cuda.is_available():
        x = x.to(device)
    flops, params = profile(model, inputs=(x,), verbose=True)

    gflops = flops / 1e9
    params_in_millions = params / 1e6
    print(f'Flops: {gflops:.3f}G')
    print(f'Params: {params_in_millions:.3f}M')


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def profile_model(cfg: cfg):
    # from omegaconf import OmegaConf
    # print(OmegaConf.to_yaml(cfg))

    # inst head.
    cfg.model.decoder.instance_head.type = "InstanceHead-v2.0-overlaps-attn"
    cfg.model.decoder.instance_head.in_channels = 256
    cfg.model.decoder.instance_head.kernel_dim = 256
    cfg.model.decoder.instance_head.num_groups = 1
    cfg.model.decoder.instance_head.num_masks = 100
    # mask branch.
    cfg.model.decoder.mask_branch.type = "MaskStackedConv" # MaskDoubleConv
    cfg.model.decoder.mask_branch.dim = 256
    # inst branch.
    cfg.model.decoder.instance_branch.type = "InstStackedConv" # InstDoubleConv
    cfg.model.decoder.instance_branch.dim = 256
    # model.
    cfg.model.type = "iaunet"
    cfg.model.encoder.out_indices = [1, 2, 3, 4]
    cfg.model.n_levels = 4
    cfg.model.decoder.num_convs = 1
    cfg.model.decoder.last_layer_only = False
    cfg.model.decoder.type = "iadecoder_overlaps"

    model = get_model(cfg)

    if torch.cuda.is_available():
        model = model.to("cuda:0")
    model.eval()

    get_flops(model)


if __name__ == "__main__":
    profile_model()
