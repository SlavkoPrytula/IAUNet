import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# from utils.coco.coco import COCO
from utils.coco.mask2coco import masks2coco
# from pycocotools.cocoeval import COCOeval

from .coco_evaluator import Evaluator
from utils.utils import nested_tensor_from_tensor_list

from configs import cfg
import matplotlib.pyplot as plt
import cv2
from os.path import join

from utils.visualise import visualize_grid_v2
from utils.utils import flatten_mask
from utils.opt.mask_nms import mask_nms
from utils.coco.coco import COCO

from utils.registry import EVALUATORS


import time
import psutil
import os
import itertools


# TODO: merge base and nms evaluators
@EVALUATORS.register(name="ExperimentalEvaluator")
class ExperimentalEvaluator(Evaluator):
    # coco_eval
    def __init__(self, cfg: cfg):
        super(ExperimentalEvaluator, self).__init__(cfg)

    def forward(self, model, dataloader):
        model.eval()

        start_time = time.time()  # Start time measurement
        start_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # Memory in MB

        gt_masks = []
        pred_masks = []
        pred_scores = []

        for step, batch in enumerate(dataloader):
            if batch is None:
                continue
            
            # prepare targets
            images = []
            targets = []

            # for target in batch:
            target = batch[0]
            target = {k: v.to(cfg.device) for k, v in target.items()}
            images.append(target["image"])
            targets.append(target)

            image = nested_tensor_from_tensor_list(images)   # (B, C, H, W)

            # predict.
            output = self.inference_single(model, image.tensors)

            pred = output

            scores = pred['pred_logits'].sigmoid()
            scores = scores[0, :, 0]

            masks_pred = pred['pred_masks'].sigmoid()
            masks_pred = masks_pred[0, ...]

            seg_masks = masks_pred > self.mask_threshold
            sum_masks = seg_masks.sum((1, 2)).float()

            # maskness scores.
            maskness_scores = (masks_pred * seg_masks.float()).sum((1, 2)) / sum_masks
            scores = maskness_scores

            scores = scores.detach().cpu().numpy()
            masks_pred = masks_pred.detach().cpu().numpy()
            masks_pred = (masks_pred > self.mask_threshold).astype(np.uint8)

            # print(len(scores))
            # masks_pred = masks_pred[scores > 0.3]
            # scores = scores[scores > 0.3]
            # print(len(scores))

            masks = target['masks']
            masks = masks.detach().cpu().numpy()

            # store data.
            gt_masks.append(masks)
            pred_masks.append(masks_pred)
            pred_scores.append(scores)

        print("Done")

        self.gt_coco = masks2coco(gt_masks)
        self.pred_coco = masks2coco(pred_masks, scores=pred_scores)
        self.evaluate(verbose=True)  # Assuming this function returns the mAP
        current_mAP = self.stats["mAP@0.5:0.95"]
        print(f"Initial mAP@0.5:0.95: {current_mAP}\n\n")


        log_folder = "log_0"
        step = 0
        improved = True
        while improved:
            improved = False
            for image_index in range(len(pred_masks)):
                for instance_index in range(50):  # Assuming 50 instances per image
                    original_score = pred_scores[image_index][instance_index]

                    for change in [-0.05, 0.05]:
                        new_score = max(min(original_score + change, 1.0), 0.0)  # Ensure score is within [0, 1]
                        pred_scores[image_index][instance_index] = new_score

                        # Calculate new mAP
                        self.pred_coco = masks2coco(pred_masks, scores=pred_scores)
                        self.evaluate(verbose=False)
                        new_mAP = self.stats["mAP@0.5:0.95"]

                        if new_mAP > current_mAP:
                            current_mAP = new_mAP
                            original_score = new_score  # Update original score to the best found score
                            improved = True
                            print(f"Updated Score for Image {image_index}, Instance {instance_index} to {new_score:.4f}, New mAP@0.5:0.95: {new_mAP:.4f}\n")
                        else:
                            # Revert the score change if no improvement
                            pred_scores[image_index][instance_index] = original_score

                print(f"Evaluating with new score for instance\n")
                self.evaluate(verbose=True)
                print("=" * 50 + "\n\n")

            print(f"Evaluating one entire set with new scores!!!\n")
            self.evaluate(verbose=True)
            print("=" * 50 + "\n\n")


            visualize_grid_v2(
                masks=pred_masks[0],
                titles=pred_scores[0],
                path=f"tools/analysis/performance/{log_folder}/pred_iter_{step}.jpg"
            )
            step += 1


            # gt_coco = COCO(self.gt_coco)
            # pred_coco = COCO(self.pred_coco)

            # for i in range(1, 7):
            #     out_file = f"tools/analysis/performance/{log_folder}/img_{i}.jpg"

            #     H, W = 512, 512
            #     image = np.zeros((H, W))

            #     fig, ax = plt.subplots(1, 2, figsize=[20, 10])
            #     fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

            #     # img1
            #     annIds  = gt_coco.getAnnIds(imgIds=[i])
            #     anns    = gt_coco.loadAnns(annIds)
            #     ax[0].imshow(image)

            #     gt_masks = gt_coco.getMasks(anns, alpha=0.5)
            #     for gt_mask in gt_masks:
            #         gt_mask = cv2.resize(gt_mask, (W, H))
            #         ax[0].imshow(gt_mask)

            #     # img2
            #     annIds  = pred_coco.getAnnIds(imgIds=[i])
            #     anns    = pred_coco.loadAnns(annIds)
            #     ax[1].imshow(image)

            #     pred_masks = gt_coco.getMasks(anns, alpha=0.5)
            #     for pred_mask in pred_masks:
            #         pred_mask = cv2.resize(pred_mask, (W, H))
            #         ax[1].imshow(pred_mask)

            #     for a in ax:
            #         a.axis('off')
            #         a.set_xlim(0, W)
            #         a.set_ylim(H, 0)

            #     fig.canvas.draw()
            #     fig.savefig(out_file, bbox_inches='tight', pad_inches=0)
            #     plt.close(fig)





        # possible_scores = [round(i, 1) for i in np.arange(0, 1.1, 0.1)]

        # improved = True
        # while improved:
        #     improved = False
        #     for image_index in range(len(pred_masks)):
        #         for instance_index in range(50):  # Assuming each image has 50 instances
        #             best_score_for_instance = pred_scores[image_index][instance_index]

        #             # Try changing the score for this instance to each possible score
        #             for new_score in possible_scores:
        #                 pred_scores[image_index][instance_index] = new_score

        #                 # Calculate new mAP
        #                 self.pred_coco = masks2coco(pred_masks, scores=pred_scores)
        #                 self.evaluate(verbose=False)
        #                 new_mAP = self.stats["mAP@0.5"]
        #                 print(f"Image {image_index}, Instance {instance_index}, New Score {new_score}, New mAP: {new_mAP}")
        #                 print()

        #                 # Check if mAP improved
        #                 if new_mAP > current_mAP:
        #                     current_mAP = new_mAP
        #                     best_score_for_instance = new_score
        #                     improved = True
        #                     print(f"Updated Score for Image {image_index}, Instance {instance_index} to {new_score}")

        #             # Update the score for the instance with the best found score
        #             pred_scores[image_index][instance_index] = best_score_for_instance


        # improved = True
        # while improved:
        #     improved = False
        #     for image_index in range(len(pred_masks)):
        #         for instance_index in range(50):  # Assuming each image has 50 instances
        #             # Try changing the score for this instance
        #             original_score = pred_scores[image_index][instance_index]
        #             pred_scores[image_index][instance_index] = 0.9 if original_score == 0.8 else 0.8

        #             # Calculate new mAP
        #             self.pred_coco = masks2coco(pred_masks, scores=pred_scores)
        #             self.evaluate(verbose=True)  # Assuming this function returns the mAP
        #             new_mAP = self.stats["mAP@0.5"]
        #             print()

        #             # Check if mAP improved
        #             if new_mAP > current_mAP:
        #                 current_mAP = new_mAP
        #                 improved = True
        #             else:
        #                 # Revert the score change
        #                 pred_scores[image_index][instance_index] = original_score

        # pos_scores = [0.8, 0.9]
        # combinations = list(itertools.product(pos_scores, repeat=50))

        # full_combinations = list(itertools.product(combinations, pred_masks))
        # print(len(full_combinations))

        # n = len(full_combinations)-6 if len(full_combinations)-6 > 0 else 1
        # for i in range(0, n, 6):
        #     _combinations = full_combinations[i:i+6]
            
        #     pred_masks = []
        #     pred_scores = []
        #     for i, (scores, masks) in enumerate(_combinations):
        #         pred_masks.append(masks)
        #         pred_scores.append(scores)

        #     # masks2coco
        #     print(len(gt_masks), len(pred_masks), len(pred_scores))
        #     self.gt_coco = masks2coco(gt_masks)
        #     self.pred_coco = masks2coco(pred_masks, scores=pred_scores)

        #     print("=" * 50)
        #     for s in pred_scores:
        #         print(s)
        #     print()

        #     self.evaluate(verbose=True)
        #     print("=" * 50)



        end_time = time.time()  # End time measurement
        end_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # Memory in MB

        time_elapsed = end_time - start_time
        memory_used = end_memory - start_memory  # Calculate additional memory consumed

        print(f"Processing Time: {time_elapsed:.2f} seconds")
        print(f"Memory Used: {memory_used:.2f} MB")

