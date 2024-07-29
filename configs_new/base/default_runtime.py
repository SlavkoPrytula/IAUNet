import os
from os.path import join
from datetime import datetime
import sys
sys.path.append("./")
from configs.utils import dict

TIME = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
JOB_ID = os.environ.get('SLURM_JOB_ID')

# Train configuration
Train = dict(
    epochs=2000,
    n_folds=5,
    size=[512, 512],
    batch_size=16,
    augment=True
)

# Valid configuration
Valid = dict(
    size=[512, 512],
    batch_size=1
)

# Model configuration
Model = dict(
    type='iaunet',
    in_channels=3,
    out_channels=1,
    num_classes=1,
    n_levels=4,
    num_convs=2,
    coord_conv=True,
    multi_level=True,
    mask_dim=256,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3, 4),
        pretrained=True
    ),
    instance_head=dict(
        type="InstanceHead-v1.1",
        in_channels=256,
        num_convs=2,
        num_classes=1,
        kernel_dim=256,
        num_masks=100,
        num_groups=1,
        activation="softmax"
    ),
    criterion=dict(
        type='SparseCriterion',
        losses=["labels", "masks"],
        weights=dict(
            labels=2.0,
            bce_masks=5.0,
            dice_masks=2.0,
            iou_masks=2.0,
            no_object_weight=0.1
        ),
        matcher=dict(
            type='PointSampleHungarianMatcher',
            cost_cls=2.0,
            cost_dice=2.0,
            cost_mask=5.0,
            cost_bbox=1.0,
            cost_giou=2.0
        ),
        num_classes=1
    ),
    evaluator=dict(
        type="MMDetDataloaderEvaluator",
        mask_thr=0.5,
        score_thr=0.1,
        nms_thr=0.5,
        metric='segm',
        classwise=True,
        outfile_prefix="results/coco"
    ),
    load_pretrained=False,
    weights="runs/[resnet_iaunet_multitask]/[worms]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[base]/[job=51037668]-[2024-04-25 16:52:20]/checkpoints/best.pth",
    save_checkpoint=True,
    save_model_files=True,
    load_from_files=False,
    model_files="/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=46036037]-[2023-08-09 13:25:51]/model_files"
)

# Dataset configuration
Dataset = dict(
    name="worms"
)

# Run configuration
Run = dict(
    runs_dir='runs',
    experiment_name=f'{"/".join(f"[{i}]" for i in Model["type"].split("-"))}',
    run_name=f'[{Model.backbone.type}]/[{Dataset["name"]}]/[{Model["instance_head"]["activation"]}_iam]/[kernel_dim={Model["instance_head"]["kernel_dim"]}]-[multi_level={Model["multi_level"]}]-[coord_conv={Model["coord_conv"]}]-[losses={Model["criterion"]["losses"]}]',
    group_name='[base]',
    exist_ok=False,
    comment="""
            - new bf dataset (brightfield_coco_v2.0)
            """
)

# Wandb configuration
Wandb = dict(
    project="IAUNet",
    group=f"{Run['experiment_name']}/{Dataset['name']}",
    name=f"{JOB_ID}"
)

# Main configuration
cfg = dict(
    model=Model,
    train=Train,
    valid=Valid,
    run=Run,
    dataset=Dataset,
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        weight_decay=0.05
    ),
    scheduler=dict(
        type="CosineAnnealingLR",
        eta_min=1e-6,
        T_max=2000*8,
        verbose=False
    ),
    seed=3407,
    device="cuda",
    verbose=True,
    wandb=Wandb
)
