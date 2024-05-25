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
    batch_size: int = 4
    augment: bool   = True
    

class Valid:
    size: int       = [512, 512]
    batch_size: int = 1
        
# TODO: easy view what file tha name is linked to        
class Model:
    arch: str         = 'sparse_seunet'
    # arch: str         = 'sparse_seunet_simplified'
    # arch: str         = 'sparse_seunet_ovlp_attn_v0'
    # arch: str         = 'sparse_seunet_add_overlaps'
    # arch: str         = 'sparse_seunet_occluder'
    # arch: str         = 'sparse_seunet_occluder_overlap'
    # arch: str         = 'sparse_seunet_occluder_gcn_sh'
    # arch: str         = 'sparse_seunet_occluder_gcn_mh'
    # arch: str         = 'sparse_seunet_occluder_ml'
    # arch: str         = 'sparse_seunet_occluder_mh'
    # arch: str         = 'sparse_seunet_occluder_gcn'
    # arch: str         = 'sparse_seunet_ml_iam'
    # arch: str         = 'test_sparse_seunet'
    # arch: str         = 'sparse_seunet_dwc'
    # arch: str         = 'sparse_seunet_gcn'
    # arch: str         = 'sparse_seunet_decoupled'
    # arch: str         = 'sparse_seunet_experimental'
    # arch: str         = 'occluders-sparse_seunet_occluder'

    # arch: str         = 'sparse_unet-deep_supervision-sparse_seunet'
    # arch: str         = 'sparse_unet-deep_supervision-sparse_seunet_localized_iam'
    # arch: str         = 'sparse_unet-deep_supervision-sparse_seunet_localized_iam_ds'

    # arch: str         = 'sparse_unet_occluder-deep_supervision-sparse_seunet'
    
    # arch: str         = 'hornet'
    # arch: str         = 'convnext'            
    # arch: str         = 'hornet_occluder'
    
    in_channels: int  = 1
    out_channels: int = 1

    num_groups: int   = 1
    num_classes: int  = 1

    num_convs: int    = 4
    n_levels: int     = 5

    # instance head.
    kernel_dim: int   = 256
    num_masks: int    = 100

    # model structure.
    coord_conv: bool  = True
    multi_level: bool = True
    
    # weights.
    load_pretrained: bool = True
    # weights: str          = "/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[rectangle_v0]/[rectangle]-[softmax_iam]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=43910856]-[2023-07-10 16:34:00]/checkpoints/best.pth"
    # weights: str          = "/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=45556519]-[2023-08-02 19:13:22]/checkpoints/best.pth"
    # weights: str          = "/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=45867499]-[2023-08-07 11:44:24]/checkpoints/best.pth"
    # weights: str          = "/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=45966022]-[2023-08-08 17:15:00]/checkpoints/best.pth"

    # weights: str          = "/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_occluder]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks', 'occluders']]/[job=46611364]-[2023-08-16 22:51:59]/checkpoints/best_v1.pth"

    # weights: str          = "/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[test]/[rectangle]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=46985110]-[2023-08-23 16:32:47]/checkpoints/best.pth"
    # weights: str          = "/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=47024359]-[2023-08-24 09:07:25]/checkpoints/best.pth"
    # weights: str           = "/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_dwc]-[512]/[brightfield]/[softmax_iam]/[multi_level=False]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=47874436]-[2023-09-07 10:46:26]/checkpoints/best.pth"
    
    # weights: str           = "/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet_occluder]-[512]/[brightfield]/[softmax_iam]/[multi_level=False]-[coord_conv=True]-[losses=['labels', 'masks', 'occluders']]/[job=48316691]-[2023-09-22 08:42:16]/checkpoints/best.pth"
    # weights: str           = "/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[occluders-sparse_seunet_occluder]-[512]/[brightfield]/[softmax_iam]/[multi_level=False]-[coord_conv=True]-[losses=['labels', 'masks', 'occluders']]/[job=48336523]-[2023-09-24 15:25:47]/checkpoints/best.pth"


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
        # losses=["labels", "masks", "occluders"], 
        weights=dict(
            labels=2.0,
            focal_masks=5.0,
            bce_masks=5.0,
            dice_masks=2.0,
            no_object_weight=0.1
        ),
        matcher=dict(
            type='HungarianMatcher',
            cost_dice=2.0,
            cost_cls=2.0,
            cost_mask=5.0
        ),
        num_classes=num_classes
    )

    # evaluator.
    evaluator=dict(
        type="DataloaderEvaluator",
        # type="DataloaderEvaluatorNMS", # nms
        # type="MemoryEfficientDataloaderEvaluator",
        mask_thr=0.5,
        cls_thr=0.5,
        score_thr=0.2,
        nms_thr=0.5,
    )


class Run:
    runs_dir: str           = 'runs'
    experiment_name: str    = f'{"/".join(f"[{i}]" for i in Model.arch.split("-"))}'
    run_name: str           = f'[LiveCell30Images]/[softmax_iam]/[kernel_dim={Model.kernel_dim}]-[multi_level={Model.multi_level}]-[coord_conv={Model.coord_conv}]-[losses={Model.criterion.losses}]/[job={JOB_ID}]-[{TIME}]'
    exist_ok: bool          = False
    comment: str            = """
                                - wd = 1e-4 (for more weight change)
                                - no overlap supression in the dataset
                                - overlaps
                                - added BatchNorm2d to stacked 3x3 convolutions 
                                - fixed sigmoid iam 
                                - new wd 
                                - removed iam matcher, now we match preds with gt and use indices from matched preds to guide correcponding iams
                              """
                                # - group iams (parallel iam convs for instance and overlap features)
        

class cfg(BaseConfig):
    model: Model           = Model
    train: Train           = Train
    valid: Valid           = Valid
    project: Project       = Project
    image: Image           = Image
    run: Run               = Run
    # dataset: str           = "brightfield"
    # dataset: str           = "rectangle"
    # dataset: str           = "synthetic_brightfield"
    # dataset: str           = "original_plus_synthetic_brightfield"

    # dataset: str           = "EVICAN2"
    
    # dataset: str           = "LiveCell"
    # dataset: str           = "LiveCell2Percent"
    dataset: str           = "LiveCell30Images"
    
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        # weight_decay=0.0001,
        weight_decay=0.05,
        # lr=5e-5,
    )
    scheduler=dict(
        type="CosineAnnealingLR",
        eta_min=1e-6,
        T_max=900*14,
        # T_0=25,
        # warmup_epochs=10
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
