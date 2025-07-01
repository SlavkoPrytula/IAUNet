from os.path import join
from pathlib import Path
import argparse

import hydra
from omegaconf import OmegaConf
from configs import cfg, experiment_name

from configs import cfg as _cfg
from utils.seed import set_seed
from models.factory import build_model
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from utils.logging.lightning_logger import PLLogger

from utils.callbacks import *
from utils.optimizers import *
from utils.schedulers import *
from evaluation import *
from models.losses import *

from dataset.dataloaders import build_loader, trivial_batch_collator
from main import build_dataset


def run(cfg: _cfg):
    pl.seed_everything(cfg.seed, workers=True)
    set_seed(cfg.seed)

    # create directories.
    cfg.run.save_dir = Path(cfg.run.save_dir)

    # set logger.
    cfg.logger.log.log_files = [str(cfg.run.save_dir / log) for log in cfg.logger.log.log_files]
    logger = PLLogger(
        name="iaunet", 
        log_files=cfg.logger.log.log_files,
        save_dir=cfg.run.save_dir,
        )

    # get dataloaders
    test_dataset = build_dataset(cfg, "test")

    test_dataloader = build_loader(test_dataset, 
                                    batch_size=cfg.dataset.test_dataset.batch_size, 
                                    num_workers=4, #cfg.trainer.num_workers, 
                                    collate_fn=trivial_batch_collator, 
                                    seed=cfg.seed, 
                                    shuffle=False)

    # build and prepare model.
    model = build_model(cfg)

    evaluators = {
        "test": {
            "coco": CocoEvaluator(cfg=cfg, dataset=test_dataset), 
        },
    }

    callbacks = {}
    # add coco evaluation callback
    coco_eval_callback = CocoEval(save_coco_vis=True,
                                  alpha=0.65, 
                                  draw_border=True, 
                                  border_size=3, 
                                  border_color='white',
                                  static_color=False, 
                                  show_img=False,
                                  save_dir=cfg.run.save_dir
                                  )
    coco_eval_callback.evaluators = evaluators
    callbacks['coco_eval'] = coco_eval_callback

    print(f"Using callbacks: {list(callbacks.keys())}")
    callbacks = list(callbacks.values())

    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.strategy,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        max_epochs=cfg.trainer.max_epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        precision=cfg.trainer.precision,
        sync_batchnorm=cfg.trainer.sync_batchnorm,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_model_summary=cfg.trainer.enable_model_summary,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
    )
    
    ckpt_path = cfg.model.ckpt_path
    trainer.test(model, dataloaders=test_dataloader, ckpt_path=ckpt_path)
    

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
    parser.add_argument('--experiment_path', type=str, default=None, help='Path to the experiment directory')
    parser.add_argument('--experiment_name', type=str, default='', help='name of the experiment')
    return parser.parse_args()


if __name__ == '__main__':
    import sys
    sys.path.append("./")
    args = parse_args()

    experiment_path = Path("runs/benchmarks_v2/[Revvity_25]/[iaunet-r50]/[iadecoder_ml_fpn_ds]/[job=58335579]-[2025-06-30 19:13:04]")

    if args.experiment_path:
        experiment_path = Path(args.experiment_path)

    cfg = get_config_from_path(experiment_path)
    print(f'>>> {cfg.dataset.name}')

    eval_cfg = OmegaConf.create({
        'model': {
            'evaluator': {
                'type': "MMDetDataloaderEvaluator",
                'mask_thr': 0.5,
                'score_thr': 0.01,
                'nms_thr': 0.8,
                'metric': 'segm',
                'classwise': True,
                'outfile_prefix': "results/coco_test",
            },
            'ckpt_path': experiment_path / "checkpoints/best.ckpt",
            'load_pretrained': False,
            'save_model_files': False,
            'load_from_files': True,
            'model_files': experiment_path / "model_files",
        },
        'run': {
            'run_name': cfg.run.run_name,
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
            "strategy": "auto",
        }
    })
    cfg = OmegaConf.merge(cfg, trainer_cfg)

    run(cfg)
