import logging
from pytorch_lightning.loggers.logger import Logger
from typing import Dict, Optional, Any
from .setup_logger import setup_logger


class PLLogger(Logger):
    """
    Enhanced PyTorch Lightning logger that wraps Python's logging.Logger.
    All metrics and hyperparameters are logged to file and stdout using logger.info().
    """

    def __init__(self, name="iaunet", log_files=None, save_dir=None, level=logging.INFO):
        super().__init__()
        self.logger = setup_logger(name=name, log_files=log_files, level=level)
        self._experiment = self.logger
        self._save_dir = save_dir

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def name(self):
        return "PLLogger"

    @property
    def version(self) -> Optional[str]:
        return None

    @property
    def experiment(self) -> Any:
        return self._experiment

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        msg = f"Step {step}: " if step is not None else ""
        msg += ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        self.logger.info(msg)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        self.logger.info(f"Hyperparameters: {params}")

    def save(self) -> None:
        pass  # Not needed for file logger

    def finalize(self, status: str) -> None:
        self.logger.info(f"Training finished with status: {status}")

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)

    def debug(self, msg: str) -> None:
        self.logger.debug(msg)

    def log(self, level, msg: str) -> None:
        self.logger.log(level, msg)

