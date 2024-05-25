import os
from os.path import join
from datetime import datetime
import sys
sys.path.append("./")
from configs.utils import BaseConfig, _dict as dict


TIME = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
JOB_ID = os.environ.get('SLURM_JOB_ID')


class Image:
    size: int   = 1080


class Project:
    home_dir: str    = '/gpfs/space/home/prytula'
    work_dir: str    = join(home_dir, 'data/mask_labels/x63_fl') # home_dir + data_dir
    project_id: str = 'project-2-at-2022-05-20-09-23-83cd15f1'


# TODO: add Visuals cfg
# visualize_gt

            
class Train:
    epochs: int     = 2000
    n_folds: int    = 5
    size: int       = [512, 512]
    batch_size: int = 16
    augment: bool   = True
    

class Valid:
    size: int       = [512, 512]
    batch_size: int = 1
        

# resnet50 + DoubleConv_v2 + num_convs=2
# num_heads=1, InstanceHead-v1.1 + IAM
class Model:
    type: str         = 'iaunet'
    # arch: str         = 'resnet_iaunet_occluders'
    
    # model structure.
    in_channels: int  = 3
    out_channels: int = 1

    num_classes: int  = 1
    n_levels: int     = 4
    num_convs: int    = 2

    coord_conv: bool  = True
    multi_level: bool = True

    # mask head.
    mask_dim: int     = 256
    # inst_dim: int     = 256

    # backbone.
    # backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3, 4),
    #     pretrained=True
    # )
    backbone=dict(
        type='SwinTransformer',
    )

    # instance head.
    instance_head=dict(
        type="InstanceHead-v1.1",
        # type="InstanceHead-v3-multiheaded",
        # type="InstanceHead-v1.2-occluders",
        in_channels=256,
        num_convs=2,
        num_classes=num_classes,
        kernel_dim=256,
        num_masks=100,
        num_groups=1,
        activation="softmax"
    )

    # activation: str   = "softmax"
    # activation: str   = "sigmoid"
    # activation: str   = "temp_softmax"
    

    # losses.
    criterion=dict(
        type='SparseCriterion',
        # losses=["masks"], 
        losses=["labels", "masks"], 
        # losses=["labels", "masks", "iou"], 
        # losses=["labels", "masks", "overlaps"], 
        # losses=["labels", "masks", "borders"], 
        # losses=["labels", "masks", "bboxes"], 
        weights=dict(
            labels=2.0,
            bce_masks=5.0,
            dice_masks=2.0,
            iou_masks=2.0,
            no_object_weight=0.1
        ),
        matcher=dict(
            # type='HungarianMatcher',
            type='PointSampleHungarianMatcher',
            cost_cls=2.0,
            cost_dice=2.0, 
            cost_mask=5.0,
            cost_bbox=1.0,
            cost_giou=2.0,
        ),
        num_classes=num_classes
    )

    # evaluator.
    evaluator=dict(
        type="MMDetDataloaderEvaluator",
        mask_thr=0.5,
        score_thr=0.1,
        nms_thr=0.5,
        metric='segm', 
        classwise=True,
        outfile_prefix="results/coco" # prefix to run dir
    )


    # weights.
    load_pretrained: bool = False
    # x1 backbone
    # weights: str          = "runs/[iaunet]/[brightfield_coco_v2.0]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=50320116]-[2024-02-11 01:15:49]/checkpoints/best.pth"
    # new
    # weights: str          = "runs/[iaunet_optim_v2]/[experimental]/[job=50432642]-[2024-02-24 13:56:31]/checkpoints/best.pth"
    # weights: str          = "runs/[iaunet_optim_v2]/[experimental]/[job=50451163]-[2024-02-27 16:08:49]/checkpoints/best.pth"
    weights: str          = "runs/[resnet_iaunet_multitask]/[worms]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[base]/[job=51037668]-[2024-04-25 16:52:20]/checkpoints/best.pth"
    
    save_checkpoint: bool  = True
    save_model_files: bool = True

    # TODO: load config from experiment + param overloading
    # experimental.
    load_from_files: bool = False
    model_files: str      = "/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=46036037]-[2023-08-09 13:25:51]/model_files"



class Dataset:
    # name: str           = "brightfield"
    # name: str           = "synthetic_brightfield"
    # name: str           = "original_plus_synthetic_brightfield"

    # name: str           = "EVICAN2"
    
    # name: str           = "LiveCell"
    # name: str           = "LiveCell2Percent"
    # name: str           = "LiveCell30Images"

    # name: str           = "YeastNet"
    # name: str           = "HuBMAP"

    # name: str           = "brightfield_coco"
    # name: str           = "brightfield_coco_v2.0"
    # name: str           = "brightfield_v2.0"

    # name: str           = "rectangle"
    name: str           = "worms"



class Run:
    runs_dir: str           = 'runs'
    experiment_name: str    = f'{"/".join(f"[{i}]" for i in Model.type.split("-"))}'
    run_name: str           = f'[{Model.backbone.type}]/[{Dataset.name}]/[{Model.instance_head.activation}_iam]/[kernel_dim={Model.instance_head.kernel_dim}]-[multi_level={Model.multi_level}]-[coord_conv={Model.coord_conv}]-[losses={Model.criterion.losses}]'
    group_name: str           = f'[base]'
    # group_name: str           = f'[experimental]'
    # group_name: str           = f'[resnet_encoder]'
    # group_name: str           = f'[ia_groups]'
    # group_name: str           = f'[occluders]'
    exist_ok: bool          = False
    comment: str            = """
                                - new bf dataset (brightfield_coco_v2.0)
                              """
        

class Wandb:
    project: str = "IAUNet"
    group: str = f"{Run.experiment_name}/{Dataset.name}"
    name: str = f"{JOB_ID}"


class cfg(BaseConfig):
    model: Model           = Model
    train: Train           = Train
    valid: Valid           = Valid
    project: Project       = Project
    image: Image           = Image
    run: Run               = Run
    dataset: Dataset       = Dataset
    
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        weight_decay=0.05,
        # weight_decay=1e-5
    )
    scheduler=dict(
        type="CosineAnnealingLR",
        eta_min=1e-6,
        T_max=2000*8,
        verbose=False
    )
    # scheduler=dict(
    #     type="MultiStepLR",
    #     milestones=[500, 800],
    #     gamma=0.1,
    #     verbose=True
    # )

    seed: int = 3407
    device: str = "cuda"
    verbose: bool = True
    wandb: Wandb = Wandb
    

cfg = cfg()
# print(cfg)

# from utils import Config
# cfg = Config(cfg().__dict__())
# print(cfg)

# print(cfg.__dict__)
# print(cfg().__dict__())

# TODO: add CosineAnnealingLR with warmup epochs
