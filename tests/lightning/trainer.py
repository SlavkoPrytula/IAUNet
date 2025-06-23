import pytorch_lightning as pl
import torch
import torch.utils
import torch.utils.data


class DummyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.layer(x).mean() * 0 + 1.23
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class MetricCheckCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        assert "train/loss" in trainer.callback_metrics
        print("train/loss in callback_metrics:", trainer.callback_metrics["train/loss"])

        pl_module.log("train/stats", 12345, on_epoch=True, prog_bar=True)


class LoggerCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print("Callback: train/stats =", trainer.callback_metrics.get("train/stats", "not logged"))



def test_callback_metrics():
    model = DummyModel()
    trainer = pl.Trainer(
        max_epochs=1,
        callbacks=[MetricCheckCallback(), LoggerCallback()],
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False
    )

    data = torch.randn(8, 2)
    dataset = torch.utils.data.TensorDataset(data, data)
    trainer.fit(model, torch.utils.data.DataLoader(dataset, batch_size=2))


if __name__ == "__main__":
    test_callback_metrics()