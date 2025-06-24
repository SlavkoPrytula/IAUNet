from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any

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
class COCODataset:
    images: str
    ann_file: str
    size: Tuple[int, int]
    batch_size = int

@dataclass
class Dataset:
    name: str
    type: str
    data_root: str
    train_dataset: COCODataset
    eval_dataset: COCODataset
    valid_dataset: COCODataset
    mean: List[float]
    std: List[float]

@dataclass
class Trainer:
    accelerator: Optional[str] = "gpu"
    devices: int = 1
    num_workers: int = 4
    max_epochs: int = 250
    check_val_every_n_epoch: int = 10
    log_every_n_steps: int = 10
    deterministic: bool = None
    benchmark: bool = None
    strategy: Optional[str] = "auto"
    precision: Optional[str] = "32-true"
    num_nodes: Optional[int] = 1
    sync_batchnorm: Optional[bool] = False
    enable_progress_bar: bool = True
    enable_model_summary: bool = True
    enable_checkpointing: bool = True
    profiler: Optional[str] = None
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = None
    gradient_clip_algorithm: Optional[str] = None


@dataclass
class Visualizer:
    type: Optional[str] = "BaseVisualizer"
    epoch_interval: Optional[int] = 10

@dataclass
class Callbacks:
    visualizer: Visualizer

@dataclass
class Matcher:
    type: Optional[str] = "PointSampleHungarianMatcher"
    cost_cls: float = 1.0
    cost_mask: float = 5.0
    cost_dice: float = 2.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0
    num_points: int = 112 * 112
    
@dataclass
class Criterion:
    type: Optional[str] = "SparseCriterion"
    losses: Optional[List[str]] = field(default_factory=lambda: ["labels", "masks"])
    weights: dict = field(default_factory=dict)
    matcher: Optional[Matcher] = None
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
class Solver:
    optimizer: Optimizer
    backbone_multiplier: float = 0.1
    embedding_weight_decay: float = 0.0

@dataclass
class Scheduler:
    type: str = "CosineAnnealingLR"
    eta_min: float = 1.0e-06
    T_max: int = 16000
    verbose: bool = False


@dataclass
class Encoder:
    type: str
    depth: int = None
    num_stages: int = None
    out_indices: tuple = None
    pretrained: bool = True


@dataclass
class Decoder:
    type: str
    num_convs: int = 2
    coord_conv: bool = True
    last_layer_only: bool = False
    mask_branch: Optional[dict] = None
    instance_branch: Optional[dict] = None
    instance_head: Optional[dict] = None
    num_classes: int = 1


@dataclass
class Model:
    type: str
    in_channels: int = 3
    num_classes: int = 1
    n_levels: int = 4

    encoder: Optional[Encoder] = None
    decoder: Optional[Decoder] = None
    criterion: Optional[Criterion] = None
    evaluator: Optional[Evaluator] = None
    solver: Optional[Solver] = None
    scheduler: Optional[Scheduler] = None
    load_pretrained: bool = False
    ckpt_path: Optional[str] = None
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