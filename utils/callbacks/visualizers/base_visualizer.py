import os
import torch
from os import makedirs

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning import LightningModule
# from configs import cfg as _cfg


class BaseVisualizer(Callback):
    def __init__(self, log_every_n_epochs=1, save_dir=None):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.save_dir = save_dir

    def setup(self, trainer, pl_module, stage=None):
        super().setup(trainer, pl_module, stage)

        if self.save_dir is None:
            if hasattr(trainer, "logger") and hasattr(trainer.logger, "save_dir"):
                self.save_dir = trainer.logger.save_dir
            else:
                self.save_dir = trainer.default_root_dir

    def plot(self, model: LightningModule, batch, split="train"):
        epoch = model.current_epoch
        if epoch % self.log_every_n_epochs != 0:
            return

        save_path = os.path.join(self.save_dir, f"{split}_visuals", f"epoch_{epoch}")
        makedirs(save_path, exist_ok=True)

        is_train = model.training
        if is_train:
            model.eval()

        with torch.no_grad():
            images, targets = model._prepare_batch(batch)
            batch = {"images": images.tensors, "targets": targets}
            output = model(batch)

        self.plot_preds(output, save_path)
        self.plot_aux_preds(output, save_path)

        if is_train:
            model.train()

    # @rank_zero_only
    # def on_train_epoch_end(self, trainer, pl_module):
    #     dataloader = trainer.train_dataloader
    #     batch = next(iter(dataloader))
    #     self.plot(pl_module, batch, split="train")

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        dataloader = trainer.val_dataloaders
        if isinstance(dataloader, (list, tuple)):
            dataloader = dataloader[0]
        batch = next(iter(dataloader))
        self.plot(pl_module, batch, split="valid")

    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module):
        dataloader = trainer.test_dataloaders
        if isinstance(dataloader, (list, tuple)):
            dataloader = dataloader[0]
        batch = next(iter(dataloader))
        self.plot(pl_module, batch, split="test")

    def plot_preds(self, output, save_path):
        self._plot_preds(output, save_path=save_path)

    def plot_aux_preds(self, output, save_path):
        """
        Aux Mask Visuals
        """
        # Aux Pred Masks.
        if "aux_outputs" in output:
            for i, aux_outputs in enumerate(output['aux_outputs']):
                self._plot_preds(aux_outputs, save_path=f"{save_path}/aux_outputs/layer_{i}")

    def _plot_preds(self, output, save_path):
        """
        Pred Mask Visuals
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

