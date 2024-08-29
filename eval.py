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

    eval_dataset = dataset(cfg, 
                            dataset_type="eval",
                            transform=valid_transforms(cfg),
                            )
    
    eval_dataloader = build_loader(eval_dataset, 
                                    batch_size=cfg.dataset.valid_dataset.batch_size, 
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


    from evaluation import OverlapIOUEvaluator, MMDetDataloaderEvaluator
    evaluators = {
        "eval": {
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
    parser.add_argument('--experiment_name', type=str, default='', help='name of the experiment')
    return parser.parse_args()


if __name__ == '__main__':
    import sys
    sys.path.append("./")
    args = parse_args()

    experiment_path = Path("runs/[resnet_iaunet_multitask]/[truncated_decoder-iadecoder_ml]/[ResNet]/[LiveCellCrop]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[InstanceHead-v1.1]/[job=51959650]-[2024-08-27 15:00:50]")
    cfg = get_config_from_path(experiment_path)
    old_dataset = cfg.dataset.name
    
    # cfg.dataset = "rectangle"
    # cfg.dataset.name = "worms"
    # cfg.dataset = "brightfield"
    # cfg.dataset = "brightfield_coco"
    # cfg.dataset = "brightfield_coco_v2.0"
    # cfg.dataset = "EVICAN2Easy"
    # cfg.dataset = "EVICAN2Medium"
    # cfg.dataset = "EVICAN2Difficult"
    # cfg.dataset = "LiveCell"
    # cfg.dataset = "LiveCell2Percent"
    cfg.dataset.name = "LiveCellCrop"
    # cfg.dataset = "YeastNet"
    # cfg.dataset = "HuBMAP"

    eval_cfg = OmegaConf.create({
        'model': {
            'evaluator': {
                'type': "MMDetDataloaderEvaluator",
                'mask_thr': 0.5,
                'score_thr': 0.1,
                'nms_thr': 0.5,
                'metric': 'segm',
                'classwise': True,
                'outfile_prefix': "results/coco",
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
            'load_pretrained': True,
            'save_model_files': False,
            'load_from_files': True,
            'model_files': experiment_path / "model_files",
        },
        'run': {
            'run_name': join(cfg.run.run_name.replace(old_dataset, cfg.dataset.name)),
            # 'save_dir': join(cfg.run.runs_dir, "evals", cfg.run.experiment_name, 
            #                  cfg.run.run_name, cfg.run.group_name, str(experiment_path).split("/")[-1]),
        },
        'dataset': {
            'valid_dataset': {
                'batch_size': 16
            },
            'eval_dataset': {
                'batch_size': 16,
            }
        }
    })
    cfg = OmegaConf.merge(cfg, eval_cfg)

    trainer_cfg = OmegaConf.create({
        "trainer": {
            "accelerator": "gpu",
            "devices": 1,
            "num_workers": 2,
            "strategy": None,
        }
    })
    cfg: _cfg = OmegaConf.merge(cfg, trainer_cfg)

    run(cfg)



# python eval.py