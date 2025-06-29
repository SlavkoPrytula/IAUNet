from pytorch_lightning import Callback
import re
from utils.registry import CALLBACKS


CALLBACKS.register(name="CSVLogger")
class CSVLogger(Callback):
    def __init__(self, save_dir=None):
        super().__init__()
        self.save_dir = save_dir

    def setup(self, trainer, pl_module, stage: str):
        super().setup(trainer, pl_module, stage)

        if self.save_dir is None:
            if hasattr(trainer, "logger") and hasattr(trainer.logger, "save_dir"):
                self.save_dir = trainer.logger.save_dir
            else:
                self.save_dir = trainer.default_root_dir
        pl_module.logger.info(f"CSVLogger save_dir: {self.save_dir}")

    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        results = trainer.callback_metrics

        results = {k: v for k, v in results.items() if "step" not in k}

        metrics_items = [(k, v) for k, v in results.items() if k.startswith("metrics/")]
        # other_items = [(k, v) for k, v in results.items() if not k.startswith("metrics/")]
        # ordered_items = metrics_items + other_items
        ordered_items = metrics_items
    
        metrics = ['epoch'] + [k.replace("metrics/", "") for k, v in ordered_items]
        vals = [current_epoch] + [self._format_value(k, v) for k, v in ordered_items]
        csv_path = self.save_dir / "results.csv"
        self._save_results_csv(metrics, vals, csv_path)

    def on_test_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        results = trainer.callback_metrics

        results = {k: v for k, v in results.items() if "step" not in k}

        metrics_items = [(k, v) for k, v in results.items() if k.startswith("metrics/")]
        # other_items = [(k, v) for k, v in results.items() if not k.startswith("metrics/")]
        # ordered_items = metrics_items + other_items
        ordered_items = metrics_items
    
        metrics = ['epoch'] + [k.replace("metrics/", "") for k, v in ordered_items]
        vals = [current_epoch] + [self._format_value(k, v) for k, v in ordered_items]
        csv_path = self.save_dir / "test_results.csv"
        self._save_results_csv(metrics, vals, csv_path)

    def _format_value(self, key, value):
        if re.search(r"loss", key, re.IGNORECASE):
            try:
                return f"{float(value):.4f}"
            except Exception:
                return str(value)
        try:
            return f"{float(value):.5g}"
        except Exception:
            return str(value)

    def _save_results_csv(self, metrics, vals, csv_path):
        col_widths = [max(len(str(m)), len(str(v))) + 2 for m, v in zip(metrics, vals)]
        header = "\t".join(f"{m:<{w}}" for m, w in zip(metrics, col_widths))
        row = "\t".join(f"{v:<{w}}" for v, w in zip(vals, col_widths))
        write_header = not csv_path.exists()
        with open(csv_path, 'a') as f:
            if write_header:
                f.write(header + '\n')
            f.write(row + '\n')
