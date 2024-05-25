import os
from os.path import join
from datetime import datetime
import inspect
import json 
import yaml
from pathlib import Path
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
    epochs: int     = 1001
    n_folds: int    = 5
    size: int       = [512, 512]
    batch_size: int = 2
    augment: bool   = True
    

class Valid:
    size: int       = [512, 512]
    batch_size: int = 1
        
# TODO: easy view what file tha name is linked to        
class Model:
    arch: str         = 'sparse_seunet'
    
    in_channels: int  = 3
    out_channels: int = 1

    num_groups: int   = 1
    num_classes: int  = 1

    num_convs: int    = 4
    n_levels: int     = 5

    # instance head.
    kernel_dim: int   = 128
    mask_dim: int     = 128
    num_masks: int    = 100

    # model structure.
    coord_conv: bool  = True
    multi_level: bool = True

    activation: str   = "softmax"
    # activation: str   = "sigmoid"
    # activation: str   = "temp_softmax"
    
    # weights.
    load_pretrained: bool = False
    weights: str          = "runs/[sparse_seunet]/[brightfield]/[sigmoid_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49001319]-[2023-11-02 09:50:32]/checkpoints/best.pth"
    
    save_checkpoint: bool  = True
    save_model_files: bool = True

    # TODO: load config from experiment + param overloading
    # experimental.
    load_from_files: bool = False
    model_files: str      = "/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=46036037]-[2023-08-09 13:25:51]/model_files"

    # losses.
    criterion=dict(
        type='SparseCriterion',
        # losses=["masks"], 
        losses=["labels", "masks"], 
        # losses=["labels", "masks", "iou"], 
        # losses=["labels", "masks", "occluders"], 
        weights=dict(
            labels=2.0,
            bce_masks=5.0,
            dice_masks=2.0,
            iou_masks=2.0, # alignment
            no_object_weight=0.1
        ),
        matcher=dict(
            type='HungarianMatcher',
            cost_dice=5.0,
            cost_cls=2.0,
            cost_mask=5.0
        ),
        num_classes=num_classes
    )

    # evaluator.
    evaluator=dict(
        # type="DataloaderEvaluator",
        # type="DataloaderEvaluatorNMS", # nms
        # type="MemoryEfficientDataloaderEvaluator",
        # type="ExperimentalEvaluator",
        type="MMDetDataloaderEvaluator",
        mask_thr=0.4,
        cls_thr=-1,
        score_thr=0.05,
        nms_thr=0.5,
    )


class Dataset:
    # name: str           = "brightfield"
    # name: str           = "rectangle"
    # name: str           = "synthetic_brightfield"
    # name: str           = "original_plus_synthetic_brightfield"

    # name: str           = "EVICAN2"
    
    # name: str           = "LiveCell"
    # name: str           = "LiveCell2Percent"
    # name: str           = "LiveCell30Images"

    # name: str           = "YeastNet"
    # name: str           = "HuBMAP"

    # name: str           = "brightfield_coco"
    name: str           = "brightfield_coco_v2.0"




class Run:
    runs_dir: str           = 'runs'
    experiment_name: str    = f'{"/".join(f"[{i}]" for i in Model.arch.split("-"))}'
    run_name: str           = f'[{Dataset.name}]/[{Model.activation}_iam]/[kernel_dim={Model.kernel_dim}]-[multi_level={Model.multi_level}]-[coord_conv={Model.coord_conv}]-[losses={Model.criterion.losses}]/[job={JOB_ID}]-[{TIME}]'
    exist_ok: bool          = False
    comment: str            = """
                                - new bf dataset (brightfield_v2.0)
                              """
        

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
    )
    scheduler=dict(
        type="CosineAnnealingLR",
        eta_min=1e-6,
        T_max=900*14,
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
    

cfg = cfg()
# print(cfg)

# from utils import Config
# cfg = Config(cfg().__dict__())
# print(cfg)

# print(cfg.__dict__)
# print(cfg().__dict__())

# TODO: add CosineAnnealingLR with warmup epochs
