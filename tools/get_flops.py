import sys
import hydra

sys.path.append("./")

from configs import cfg
from models.build_model import get_model
from thop import profile
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table



def compute_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_flops_counter(model, input_tensor):
    flop_counter = FlopCountAnalysis(model, input_tensor)
    return flop_counter

def compute_flops(model, input_tensor):
    flop_counter = get_flops_counter(model, input_tensor)
    return flop_counter.total() * 2

def _profile(model, input, max_depth=2):
    params = compute_params(model)
    flops = compute_flops(model, input)

    tab = flop_count_table(get_flops_counter(model, input), max_depth=max_depth)
    print()
    print(tab)
    print()

    torch.cuda.empty_cache()

    return flops, params


def get_flops(model, device="cuda:0"):
    # get-flops
    model.eval()
    
    x = torch.randn(1, 3, 512, 512)
    if torch.cuda.is_available():
        x = x.to(device)
    # flops, params = profile(model, inputs=(x,), verbose=True)
    flops, params = _profile(model, x)

    gflops = flops / 1e9
    params = params / 1e6
    print(f'Flops: {gflops:.3f}G')
    print(f'Params: {params:.3f}M')
    print()


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def profile_model(cfg: cfg):

    if True:
        # transformer decoder config.
        cfg.model.decoder.num_queries = 100
        cfg.model.decoder.num_classes = 1
        cfg.model.decoder.hidden_dim = 256
        cfg.model.decoder.nheads = 8
        cfg.model.decoder.dec_layers = 1
        cfg.model.decoder.dropout = 0.0
        cfg.model.decoder.pre_norm = False
        cfg.model.decoder.dim_feedforward = 1024

        # pixel decoder config.
        cfg.model.decoder.mask_branch.type = "MaskStackedConv" # MaskDoubleConv, MaskStackedConv
        cfg.model.decoder.mask_branch.dim = 256
        
        # model.
        cfg.model.type = "iaunet_v2"
        cfg.model.decoder.type = "iadecoder_ml_fpn"
        cfg.model.decoder.n_levels = 4

    model = get_model(cfg)

    if torch.cuda.is_available():
        model = model.to("cuda:0")

    get_flops(model)


if __name__ == "__main__":
    profile_model()