from dataclasses import dataclass, field
from typing import List, Optional, Any

@dataclass
class Log:
    name: Optional[str] = 'iaunet'
    level: Optional[str] = "INFO"
    log_files: Optional[List[str]] = None

@dataclass
class WandbLogger:
    project: str = "IAUNet"
    group: Optional[str] = None
    name: Optional[str] = None

@dataclass
class Logger:
    log: Log
    wandb: WandbLogger

@dataclass
class Run:
    runs_dir: str = 'runs'
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    group_name: Optional[str] = None
    save_dir: Optional[str] = None


@dataclass
class Dataset:
    name: str
    type: str
    data_root: str
    train_dataset: dict
    eval_dataset: dict
    valid_dataset: dict

@dataclass
class Trainer:
    accelerator: Optional[str] = "gpu"
    devices: int = 1
    max_epochs: int = 250
    check_val_every_n_epoch: int = 10
    deterministic: bool = False
    strategy: Optional[str] = None
    num_nodes: Optional[int] = 1
    sync_batchnorm: Optional[bool] = False

@dataclass
class Visualizer:
    type: Optional[str] = "BaseVisualizer"
    epoch_interval: Optional[int] = 10

@dataclass
class Callbacks:
    visualizer: Visualizer

@dataclass
class Encoder:
    type: str
    depth: Optional[int] = None
    num_stages: Optional[int] = None
    out_indices: Optional[tuple] = None
    pretrained: Optional[bool] = True

@dataclass
class Criterion:
    type: Optional[str] = "SparseCriterion"
    losses: Optional[List[str]] = field(default_factory=lambda: ["labels", "masks"])
    weights: dict = field(default_factory=dict)
    matcher: dict = field(default_factory=dict)
    num_classes: int = 1

@dataclass
class Evaluator:
    type: str = "MMDetDataloaderEvaluator"
    mask_thr: float = 0.5
    score_thr: float = 0.1
    nms_thr: float = 0.5
    metric: str = "segm"
    classwise: bool = True
    outfile_prefix: str = "results/coco"

@dataclass
class Optimizer:
    type: str = "AdamW"
    lr: float = 0.0001
    weight_decay: float = 0.05

@dataclass
class Scheduler:
    type: str = "CosineAnnealingLR"
    eta_min: float = 1.0e-06
    T_max: int = 16000
    verbose: bool = False

@dataclass
class Model:
    type: str
    in_channels: int = 3
    out_channels: int = 1
    num_classes: int = 1
    n_levels: int = 4
    num_convs: int = 4
    coord_conv: bool = True
    multi_level: bool = True
    mask_dim: int = 256
    encoder: Optional[Encoder] = None
    instance_head: Optional[dict] = None
    criterion: Optional[Criterion] = None
    evaluator: Optional[Evaluator] = None
    optimizer: Optional[Optimizer] = None
    scheduler: Optional[Scheduler] = None
    load_pretrained: bool = False
    weights: str = ''
    save_checkpoint: bool = True
    save_model_files: bool = True
    load_from_files: bool = False
    model_files: str = ''

@dataclass
class cfg:
    dataset: Dataset
    logger: Logger
    wandb: WandbLogger
    run: Run
    trainer: Trainer
    callbacks: Callbacks
    model: Model
    seed: int = 3407
    
    train: bool = True
    test: bool = True

    job_id: Optional[str] = ''
    run_id: Optional[str] = ''


from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="base_config", node=cfg)

# hydra_core-1.3.2