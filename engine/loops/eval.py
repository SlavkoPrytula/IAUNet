from os import makedirs
from os.path import join
from typing import Dict
import wandb

import torch
import os
import json
from itertools import islice
from visualizations.coco_vis import save_coco_vis
from models import load_weights

from evaluation.evaluators import BaseEvaluator
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
        evaluators: Dict[str, BaseEvaluator], 
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
        self.evaluators = evaluators.get('eval')
        self.eval_dir = self.cfg.run.save_dir / 'eval'


    def run(self):
        self.model.eval()

        if not os.path.exists(self.eval_dir):
            self.logger.info(f'Saving eval results at: {self.eval_dir}')
            makedirs(self.eval_dir, exist_ok=True)
            makedirs(self.eval_dir / 'results', exist_ok=True)
            makedirs(self.eval_dir / 'visuals', exist_ok=True)
        
        self.logger.info('TestLoop')
        for evaluator_name in self.evaluators:
            print(f"Evaluating {evaluator_name} subset...")
            evaluator = self.evaluators[evaluator_name]
            evaluator.model = self.model
            evaluator(self.dataloader)
            evaluator.evaluate(verbose=True)
        
            # save results.
            stats = evaluator.stats
            results_file = self.eval_dir / 'results' / 'evaluation_results.json'
            # dataset_name = self.dataloader.dataset.name
            dataset_name = self.cfg.dataset.name
            dataset_path = self.dataloader.dataset.ann_file

            results = self.load_results(results_file)
            results = self.update_results(results, dataset_name, stats, dataset_path)
            self.save_results(results_file, results)

            if "coco" in evaluator_name:
                # plot results.
                gt_coco = evaluator.gt_coco
                pred_coco = evaluator.pred_coco

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


