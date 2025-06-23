from pytorch_lightning import Callback
from utils.flops_utils import get_flops
from utils.registry import CALLBACKS


@CALLBACKS.register(name="FlopsLogger")
class FlopsLogger(Callback):
    def __init__(self, input_size=(1, 3, 512, 512), device="cuda:0", max_depth=2):
        super().__init__()
        self.input_size = input_size
        self.device = device
        self.max_depth = max_depth

    def on_fit_start(self, trainer, pl_module):
        get_flops(pl_module, device=self.device, input_size=self.input_size, max_depth=self.max_depth)