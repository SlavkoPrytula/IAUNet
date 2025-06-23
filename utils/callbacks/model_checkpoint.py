import os
from pytorch_lightning.callbacks import ModelCheckpoint as _ModelCheckpoint
from utils.registry import CALLBACKS


@CALLBACKS.register(name="ModelCheckpoint")
class ModelCheckpoint(_ModelCheckpoint):
    def __init__(self, dirpath=None, monitor=None, mode="min", save_top_k=1, 
                 save_last=True, filename="best", **kwargs):
        super().__init__(
            dirpath=dirpath,
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=save_last,
            filename=filename,
            **kwargs
        )

    def setup(self, trainer, pl_module, stage: str):
        super().setup(trainer, pl_module, stage)

        if hasattr(trainer, "logger") and hasattr(trainer.logger, "save_dir"):
            self.dirpath = trainer.logger.save_dir
        else:
            self.dirpath = trainer.default_root_dir
        self.dirpath = os.path.join(self.dirpath, "checkpoints")
