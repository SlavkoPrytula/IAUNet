from os import makedirs
from os.path import join
import wandb

from itertools import islice
from visualizations.coco_vis import save_coco_vis
from utils.utils import nested_tensor_from_tensor_list

from utils.evaluate.coco_evaluator import Evaluator
from configs import cfg
from .base import BaseLoop


class ValidLoop(BaseLoop):
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

    def run(self):
        self.model.eval()
        dataset_size = 0
        running_loss = 0.0
        results = {}

        self.trigger_callbacks('on_valid_epoch_start', trainer=self.trainer, cfg=self.cfg, epoch=self.epoch)
        
        self.logger.info('Loss/Valid')
        for step, batch in enumerate(self.dataloader):
            if batch is None:
                continue
            
            # prepare targets
            images = []
            targets = []
            for i in range(len(batch)):
                target = batch[i]

                ignore = ["img_id", "img_path", "ori_shape", "file_name", "coco_id"]
                target = {k: v.to(self.device) if k not in ignore else v 
                        for k, v in target.items()}
                images.append(target["image"])
                targets.append(target)
                
            images = nested_tensor_from_tensor_list(images)   # (B, C, H, W)
            batch_size = images.tensors.size(0)
            
            output = self.model(images.tensors)
            output["img_id"] = target["img_id"]
            output["ori_shape"] = target["ori_shape"]

            # evaluator.
            for evaluator_name in self.evaluators:
                evaluator = self.evaluators[evaluator_name]
                evaluator.process(output)
            
            # get losses.
            loss_dict = self.criterion(output, targets, 
                                       [self.cfg.dataset.valid_dataset.size], 
                                       epoch=self.epoch)
            loss = sum(loss_dict.values())
            
            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size
            epoch_loss = running_loss / dataset_size
            self.trainer.loss = epoch_loss

            # for callback in self.callbacks:
            #     callback.on_valid_batch_end(step=step, epoch=epoch, loss=epoch_loss)
            # # loss_callback.on_valid_batch_end(step, epoch, epoch_loss)
            
            self.trigger_callbacks('on_valid_batch_end', trainer=self.trainer, cfg=self.cfg, batch=step)

        print()
        for l in loss_dict:
            self.logger.info(f'{l}: {loss_dict[l]}')
        print()
        
        
        # TODO: this should go into callback
        # NOTE: evaluation is now done every n epochs, this is not needed
        # check cfg.trainer.check_val_every_n_epoch
        makedirs(join(self.cfg.run.save_dir, 'train_visuals', f'epoch_{self.epoch}', 'results'), exist_ok=True)
        
        # evaluate.
        for evaluator_name in self.evaluators:
            print(f"Evaluating {evaluator_name} subset...")
            evaluator = self.evaluators[evaluator_name]
            evaluator.evaluate(verbose=True)
            stats = evaluator.stats

            if evaluator_name in ["valid", "eval"]:
                results.update(stats)  

                # TODO: check from cfg
                if wandb.run is not None:
                    for s in stats:
                        # wandb results.
                        wandb.log({f"metrics/{s}": stats[s]})

            # plot results.
            gt_coco = evaluator.gt_coco
            pred_coco = evaluator.pred_coco

            for batch in islice(self.dataloader, 2):
                targets = batch[0]
                
                img = targets["image"][0]
                fname = targets["file_name"]
                idx = targets["coco_id"]
                H, W = targets["ori_shape"]
                
                save_coco_vis(img, gt_coco, pred_coco, idx, shape=[H, W], 
                            path=f'{self.cfg.run.save_dir}/train_visuals/epoch_{self.epoch}/results/pred_[{evaluator_name}]_[{fname}].jpg')
                    

        # logging results.
        results["loss_valid"] = epoch_loss
        for l in loss_dict:
            results[f"{l}_valid"] = loss_dict[l]
            
        # wandb results.
        # TODO: check from cfg
        if wandb.run is not None:
            wandb.log({f"valid/loss_valid": epoch_loss})
            for l in loss_dict:
                wandb.log({f"valid/{l}_valid": loss_dict[l]})


        self.trainer.output = output
        self.trigger_callbacks('on_valid_epoch_end', trainer=self.trainer, cfg=self.cfg, epoch=self.epoch)
        
        return results
