from os import makedirs
from os.path import join
import wandb

import os
import json
from itertools import islice
from visualizations.coco_vis import save_coco_vis
from models import load_weights

from utils.evaluate.coco_evaluator import Evaluator
from configs import cfg
from .base import BaseLoop


class EvalLoop(BaseLoop):
    def __init__(
        self, 
        cfg: cfg, 
        model, 
        criterion, 
        dataloader, 
        device, 
        logger, 
        callbacks,
        evaluators, 
    ):
        super().__init__(
            cfg, 
            model, 
            criterion, 
            dataloader, 
            device, 
            logger, 
            callbacks,
            evaluators
        )
        self.evaluators = evaluators
        self.total_steps = len(self.dataloader)

        # look here
        self.evaluator = self.evaluators[0]
        self.eval_dir = self.cfg.run.save_dir / 'eval'


    def run(self):
        model = load_weights(model, weights_path=cfg.model.weights)
        model.eval()

        if not os.path.exists(self.eval_dir):
            self.logger.info(f'Saving eval results at: {self.eval_dir}')
            makedirs(self.eval_dir, exist_ok=True)
            makedirs(self.eval_dir / 'results', exist_ok=True)
            makedirs(self.eval_dir / 'visuals', exist_ok=True)
        
        self.evaluator(self.dataloader)
        self.evaluator.evaluate(verbose=True)
        
        # save results.
        stats = self.evaluator.stats
        results_file = self.eval_dir / 'results' / 'evaluation_results.json'
        dataset_name = self.cfg.dataset.name
        dataset_path = self.cfg.dataset.eval_dataset.ann_file

        results = self.load_results(results_file)
        results = self.update_results(results, dataset_name, stats, dataset_path)
        self.save_results(results_file, results)

        # plot results.
        gt_coco = self.evaluator.gt_coco
        pred_coco = self.evaluator.pred_coco

        # TODO: 2config : Visualizations {n_samples: int = 5}
        # n_samples = len(valid_dataset)
        n_samples = 6
        for batch in islice(self.dataloader, n_samples):
            targets = batch[0]
            
            img = targets["image"][0]
            fname = targets["file_name"]
            idx = targets["coco_id"]
            H, W = targets["ori_shape"]
            out_file = join(self.eval_dir, 'visuals', f'{fname}.jpg')

            save_coco_vis(img, gt_coco, pred_coco, idx, shape=[H, W], path=out_file)

    @staticmethod
    def load_results(file_path):
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r') as file:
                    content = file.read().strip()
                    if content:
                        return json.loads(content)
                    else:
                        return {}
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def save_results(file_path, results):
        sorted_results = {}
        for dataset_name in sorted(results):
            sorted_results[dataset_name] = {}
            for dataset_path in sorted(results[dataset_name]):
                sorted_results[dataset_name][dataset_path] = results[dataset_name][dataset_path]

        with open(file_path, 'w') as file:
            json.dump(sorted_results, file, indent=4)

    @staticmethod
    def update_results(existing_results, dataset_name, new_results, dataset_path):
        if dataset_name not in existing_results:
            existing_results[dataset_name] = {}

        existing_results[dataset_name][dataset_path] = new_results
        return existing_results
