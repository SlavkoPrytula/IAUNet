from os.path import join
from pathlib import Path
import argparse

import hydra
from omegaconf import OmegaConf
from configs import cfg, experiment_name

from configs import cfg as _cfg
from models.build_model import build_model
from engine.trainer import Trainer
from utils.seed import set_seed
from utils.logging import setup_logger
from utils.augmentations import train_transforms, valid_transforms

from evaluation import *
from utils.optimizers import *
from utils.schedulers import *
from models.seg.loss import *
from models.seg.matcher import *

from dataset.dataloaders import build_loader, trivial_batch_collator
from utils.registry import DATASETS, EVALUATORS



def run(cfg: _cfg):
    # create directories.
    cfg.run.save_dir = Path(cfg.run.save_dir)

    # set logger.
    logger = setup_logger(
        name=cfg.logger.log.name, 
        log_files=cfg.logger.log.log_files
        )

    # set seed for reproducibility.
    set_seed(cfg.seed)

    # get dataloaders
    dataset = DATASETS.get(cfg.dataset.type)
    print()

    eval_dataset = dataset(cfg, 
                           dataset_type="eval",
                           transform=valid_transforms(cfg))
    
    eval_dataloader = build_loader(eval_dataset, 
                                   batch_size=cfg.dataset.eval_dataset.batch_size, 
                                   num_workers=cfg.trainer.num_workers, 
                                   collate_fn=trivial_batch_collator, 
                                   seed=cfg.seed)

    # build and prepare model.
    model = build_model(cfg)
    model.eval()

    if cfg.trainer.accelerator == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)


    from evaluation import OverlapIOUEvaluator, MMDetDataloaderEvaluator, AnalysisMMDetDataloaderEvaluator
    evaluators = {
        "eval": {
            # "coco": AnalysisMMDetDataloaderEvaluator(cfg=cfg, model=model, dataset=eval_dataset), 
            "coco": MMDetDataloaderEvaluator(cfg=cfg, model=model, dataset=eval_dataset), 
            # "overlap_iou": OverlapIOUEvaluator(cfg=cfg, model=model, dataset=eval_dataset)
        },
    }

    trainer = Trainer(cfg, model, 
                      criterion=None, 
                      train_dataloader=None, 
                      valid_dataloader=None,
                      eval_dataloader=eval_dataloader,
                      optimizer=None, 
                      scheduler=None,
                      evaluators=evaluators,
                      callbacks=None,
                      logger=logger,
                      rank=None,
                      strategy=cfg.trainer.get('strategy'),
                      sync_batchnorm=cfg.trainer.get('sync_batchnorm')
                    )

    trainer.test()
    

def get_config_from_path(path: str) -> _cfg:
    try:
        config_path = join(path, "config_files", "train.yaml")
        config = OmegaConf.load(config_path)
    except Exception as e:
        print(f"Error loading configuration from {path}: {e}")
        config = _cfg  # fallback to default
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Run evaluation with IAUNet')
    parser.add_argument('experiment_path', type=str, nargs='?', default=None, help='Path to the experiment directory')
    parser.add_argument('--experiment_name', type=str, default='', help='name of the experiment')
    return parser.parse_args()


if __name__ == '__main__':
    import sys
    sys.path.append("./")
    args = parse_args()

    # pixel decoder.
    # --------------
    # [iadecoder_ml]
    # experiment_path = Path("runs/ablations/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml]/[InstanceHead-v2.2.1-dual-update]/[job=52560796]-[2024-11-06 12:18:01]")
    # [iadecoder_ml_fpn]
    # experiment_path = Path("runs/ablations/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn]/[InstanceHead-v2.2.1-dual-update]/[job=52560797]-[2024-11-06 12:18:01]")
    # [iadecoder_ml_fpn_add_skip]
    # experiment_path = Path("runs/ablations/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn_add_skip]/[InstanceHead-v2.2.1-dual-update]/[job=52560798]-[2024-11-06 12:18:01]")

    # [iadecoder_ml_fpn_no_mask_branch]
    # experiment_path = Path("runs/ablations/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn_no_mask_branch]/[InstanceHead-v2.2.1-dual-update]/[job=52577561]-[2024-11-10 01:10:08]")
    # [iadecoder_ml_fpn_no_inst_branch]
    # experiment_path = Path("runs/ablations/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn_no_inst_branch]/[InstanceHead-v2.2.1-dual-update]/[job=52577560]-[2024-11-10 01:11:42]")


    # transformer decoder.
    # --------------
    # [removed-mask-feats]
    # experiment_path = Path("runs/ablations/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn]/[InstanceHead-v2.2.a-removed-mask-feats]/[job=52560800]-[2024-11-06 12:17:46]")
    # [removed-inst-feats]
    # experiment_path = Path("runs/ablations/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn]/[InstanceHead-v2.2.a-removed-inst-feats]/[job=52560799]-[2024-11-06 12:18:01]")
    
    # [no-guided-query]
    # experiment_path = Path("runs/ablations/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn]/[InstanceHead-v2.2.a-no-guided-query]/[job=52577565]-[2024-11-10 01:31:45]")
    # [no-support-query]
    # experiment_path = Path("runs/ablations/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn]/[InstanceHead-v2.2.a-no-support-query]/[job=52577562]-[2024-11-10 01:10:08]")

    # [swin]
    # experiment_path = Path("runs/experiments_v2/[ISBI2014]/[iaunet-r50]/[iadecoder_ml_fpn]/[job=53670380]-[2025-02-09 14:23:05]")
    experiment_path = Path("runs/experiments_v2/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn]/[experimental]/[deep_supervision]/[job=9766337]-[2025-03-04 18:38:18]")



    if args.experiment_path:
        experiment_path = Path(args.experiment_path)

    cfg = get_config_from_path(experiment_path)
    old_dataset = cfg.dataset.name
    
    # cfg.dataset = "rectangle"
    # cfg.dataset.name = "worms"
    # cfg.dataset = "brightfield"
    # cfg.dataset = "brightfield_coco"
    # cfg.dataset = "brightfield_coco_v2.0"
    
    # cfg.dataset.name = "EVICAN2_Easy"
    # cfg.dataset.name = "EVICAN2_Medium"
    # cfg.dataset.name = "EVICAN2_Difficult"

    # cfg.dataset = "LiveCell"
    cfg.dataset.name = "LiveCellCrop"

    # cfg.dataset.name = "ISBI2014"

    # cfg.dataset.name = "Revvity_25"
    
    # cfg.dataset.name = "NeurlPS22_CellSeg"
    # cfg.dataset.name = "YeastNet"
    # cfg.dataset.name = "HuBMAP"

    # cfg.dataset.name = "cellpainting_gallery"

    eval_cfg = OmegaConf.create({
        'model': {
            'evaluator': {
                'type': "MMDetDataloaderEvaluator",
                'mask_thr': 0.5,
                'score_thr': 0.1,
                'nms_thr': 0.5,
                'metric': 'segm',
                'classwise': True,
                'outfile_prefix': "eval/results/coco",
            },
            'criterion': {
                'matcher': {
                    'type': 'HungarianMatcher',
                    'cost_dice': 2.0,
                    'cost_cls': 1.0,
                    'cost_mask': 5.0,
                }
            },
            'weights': experiment_path / "checkpoints/best.pth",
            'load_pretrained': False,
            'save_model_files': False,
            'load_from_files': True,
            'model_files': experiment_path / "model_files",
        },
        'run': {
            'run_name': join(cfg.run.run_name.replace(old_dataset, cfg.dataset.name)),
            'save_dir': experiment_path,
        },
        'dataset': {
            'valid_dataset': {
                'size': [512, 512],
                'batch_size': 4
            },
            'eval_dataset': {
                'batch_size': 4,
            }
        }
    })
    cfg = OmegaConf.merge(cfg, eval_cfg)

    trainer_cfg = OmegaConf.create({
        "trainer": {
            "accelerator": "gpu",
            "devices": 1,
            "num_workers": 4,
            "strategy": None,
        }
    })
    cfg: _cfg = OmegaConf.merge(cfg, trainer_cfg)

    run(cfg)



# python eval.py