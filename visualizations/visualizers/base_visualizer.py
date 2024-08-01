from os import makedirs
from os.path import join

import sys
sys.path.append("./")

from configs import cfg
from utils.registry import VISUALIZERS


@VISUALIZERS.register(name="BaseVisualizer")
class BaseVisualizer:
    def __init__(self, configs: dict = None, epoch_interval=None, **kwargs):
        self.visualizers = {}
        if configs:
            for name, cfg in configs.items():
                self.visualizers[name] = VISUALIZERS.build(cfg)
        self.cfg = None
        self.output = None
        self.save_dir = None
        self.save_path = None
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, cfg: cfg, epoch: int, output, **kwargs):
        if epoch % self.epoch_interval == 0:
            self.cfg = cfg
            self.output = output
            self.save_dir = cfg.save_dir
            self.save_path = join(cfg.save_dir, 'train_visuals', f'epoch_{epoch}')
            makedirs(self.save_path, exist_ok=True)

            self.plot(cfg, output, self.save_path)
    
    def plot(self, cfg, output, save_path):
        for name, visualizer in self.visualizers.items():
            visualizer.plot(cfg, output, save_path)


    