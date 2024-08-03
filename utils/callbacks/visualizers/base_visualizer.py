from os import makedirs
from os.path import join

from utils.callbacks import Callback
from configs import cfg
from utils.registry import CALLBACKS


@CALLBACKS.register(name="BaseVisualizer")
class BaseVisualizer(Callback):
    def __init__(self, epoch_interval=None, **configs):
        self.visualizers = {}
        if configs:
            for name, cfg in configs.items():
                self.visualizers[name] = CALLBACKS.build(cfg)
        self.cfg = None
        self.output = None
        self.save_dir = None
        self.save_path = None
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, cfg: cfg, epoch: int, **kwargs):
        if epoch % self.epoch_interval == 0:
            self.cfg = cfg
            self.output = trainer.output
            self.save_dir = cfg.run.save_dir
            self.save_path = join(cfg.run.save_dir, 'train_visuals', f'epoch_{epoch}')
            makedirs(self.save_path, exist_ok=True)

            self.plot(cfg, self.output, self.save_path)
    
    def plot(self, cfg, output, save_path):
        for name, visualizer in self.visualizers.items():
            visualizer.plot(cfg, output, save_path)


    