import torch
from os.path import join
from configs import cfg
from tqdm import tqdm
import torch.nn.functional as F

from ..coco_evaluator import COCOEvaluator
from utils.utils import nested_tensor_from_tensor_list
from utils.opt.mask_nms import mask_nms
from utils.registry import EVALUATORS, DATASETS
from evaluation.mmdet import CocoMetric

from utils.common.decorators import timeit_evaluator, memory_evaluator

import os
import numpy as np
import matplotlib.pyplot as plt



def save_image_and_masks(image, masks, iams, img_id, step, shape, alpha=0.5, draw_border=True, dpi=300):
    if step < 5:
        return
    
    if img_id == 1:
        return
    
    output_dir = f'./temp/image_{img_id}'
    os.makedirs(output_dir, exist_ok=True)

    img_path = os.path.join(output_dir, f'image_{img_id}.png')
    fig, ax = plt.subplots(1, 1, figsize=[20, 20])
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    plt.savefig(img_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    N = masks.shape[0]
    for i in tqdm(range(N)):
        output_dir = f'./temp/image_{img_id}/{img_id}_mask_{i}'
        os.makedirs(output_dir, exist_ok=True)

        mask_path_random_c = os.path.join(output_dir, f'image_{img_id}_mask_{i}_random_color.png')
        mask_path_static_c = os.path.join(output_dir, f'image_{img_id}_mask_{i}_static_color.png')
        iam_path_sigmoid = os.path.join(output_dir, f'image_{img_id}_iam_{i}_sigmoid.png')
        iam_path_softmax = os.path.join(output_dir, f'image_{img_id}_iam_{i}_softmax.png')

        visualize_single_mask_on_image(image, masks[i], shape, img_id, i, mask_path_random_c, 
                                       alpha=alpha, draw_border=draw_border, static_color=False)
        
        visualize_single_mask_on_image(image, masks[i], shape, img_id, i, mask_path_static_c, 
                                       alpha=alpha, draw_border=draw_border, static_color=True)


        # visualize iams.
        if isinstance(iams, np.ndarray):
            iams = torch.tensor(iams, dtype=torch.float32)

        # downsampled_shape = [shape[0] // 16, shape[1] // 16]
        # iams = F.interpolate(iams.unsqueeze(0), size=downsampled_shape, 
        #                 mode="bilinear", align_corners=False).squeeze(0)
        
        N, H, W = iams.shape
        _iams = iams.clone()
        _iams = F.softmax(_iams.view(N, -1), dim=-1)
        vis_preds_iams = _iams.view(N, H, W)
        plt.imsave(iam_path_softmax, vis_preds_iams[i], cmap='jet')


        vis_preds_iams = iams.clone().sigmoid()
        plt.imsave(iam_path_sigmoid, vis_preds_iams[i], cmap='jet')

        if i == 10:
            raise

    if img_id >= 5:
        raise



def save_attn(attn, img_id, step, shape, alpha=0.5, draw_border=True, dpi=300):
    # attn: (N, H, W), N=200 - num_queries, H, W - height, width
    if step < 2:
        return
    
    output_dir = f'./temp/image_{img_id}/attn'
    os.makedirs(output_dir, exist_ok=True)

    N = attn.shape[0]
    for i in tqdm(range(N)):
        attn_path = os.path.join(output_dir, f'query_attn_{i}.png')
        
        plt.imsave(attn_path, attn[i])

    if step > 5:
        return


# for self-attention
# def save_attn(attn, img_id, step, shape, alpha=0.5, draw_border=True, dpi=300):
#     # attn: (N, H, W), N=200 - num_queries, H, W - height, width
#     if step < 2:
#         return
    
#     output_dir = f'./temp/image_{img_id}/sa-attn'
#     os.makedirs(output_dir, exist_ok=True)

#     attn_path = os.path.join(output_dir, f'query_attn.png')
#     plt.imsave(attn_path, attn)

#     if step > 5:
#         raise




from visualizations.coco_vis import getNPMasks, _visualize_masks

def visualize_single_mask_on_image(img, mask, shape, img_id, mask_idx, path, 
                                   alpha=0.5, draw_border=True, static_color=False, dpi=300):
    """
    Draws a single mask on an image and saves the resulting image.
    """
    if isinstance(img, np.ndarray):
        img = torch.tensor(img, dtype=torch.float32)

    if isinstance(mask, np.ndarray):
        mask = torch.tensor(mask, dtype=torch.float32)

    img = F.interpolate(img.unsqueeze(0).unsqueeze(0), size=shape, 
                        mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=shape, 
                         mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    
    fig, ax = plt.subplots(1, 1, figsize=[20, 20])
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    ax.imshow(img, cmap='gray')

    colored_mask = getNPMasks(mask.unsqueeze(0), shape, alpha=alpha, static_color=static_color)
    _visualize_masks(ax, colored_mask, draw_border=draw_border)

    ax.axis('off')
    ax.set_xlim(0, shape[1])
    ax.set_ylim(shape[0]-1, 0)

    plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)




def plot_pred_iam_set(image, masks, iams, img_id, step, shape, alpha=0.5, draw_border=True, dpi=300):
    if step < 5:
        return
    
    output_dir = f'./temp/image_{img_id}'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'image_{img_id}_pred_iam_set.png')

    # preprocess.
    if isinstance(iams, np.ndarray):
        iams = torch.tensor(iams, dtype=torch.float32)

    iams = iams.clone().sigmoid()
    

    num_images = [1, 2, 14, 7]
    # num_images = np.arange(10, 20, 1)

    fig, axes = plt.subplots(2, len(num_images), figsize=(20, 10))
    fig.subplots_adjust(wspace=0.05, hspace=0.01, left=0, right=1, bottom=0, top=1)

    for i, idx in enumerate(num_images):
        mask = masks[idx]
        iam = iams[idx]

        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)
        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, dtype=torch.float32)
        if isinstance(iam, np.ndarray):
            iam = torch.tensor(iam, dtype=torch.float32)

        image = F.interpolate(image.unsqueeze(0).unsqueeze(0), size=shape, 
                              mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=shape, 
                             mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
        iam = F.interpolate(iam.unsqueeze(0).unsqueeze(0), size=shape, 
                            mode="bilinear", align_corners=False).squeeze(0).squeeze(0)

        # Top row: iams
        axes[0, i].imshow(iam, cmap='jet')
        axes[0, i].axis('off')

        # Bottom row: image with mask overlay
        axes[1, i].imshow(image, cmap='gray')
        colored_mask = getNPMasks(mask.unsqueeze(0), shape, alpha=alpha, static_color=True)
        _visualize_masks(axes[1, i], colored_mask, draw_border=draw_border)
        axes[1, i].axis('off')

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_attn_set(attn, img_id, step, shape, dpi=300):
    if step < 5:
        return
    
    output_dir = f'./temp/image_{img_id}'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'image_{img_id}_attn_set.png')

    if isinstance(attn, np.ndarray):
        attn = torch.tensor(attn, dtype=torch.float32)

    num_images = [0, 1, 2]
    labels = ['(a)', '(b)', '(c)']

    fig, axes = plt.subplots(1, len(num_images), figsize=(20, 10))
    fig.subplots_adjust(wspace=0.05, hspace=0.01, left=0, right=1, bottom=0, top=1)

    print(attn.shape)
    for i, idx in enumerate(num_images):
        attn_map = attn[idx]

        if isinstance(attn_map, np.ndarray):
            attn_map = torch.tensor(attn_map, dtype=torch.float32)

        attn_map = F.interpolate(attn_map.unsqueeze(0).unsqueeze(0), size=shape, 
                                 mode="bilinear", align_corners=False).squeeze(0).squeeze(0)

        axes[i].imshow(attn_map, cmap='viridis')
        axes[i].axis('off')
        axes[i].text(0.5, -0.05, labels[i], ha='center', va='top', 
                     transform=axes[i].transAxes, fontsize=25)

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def remove_padding(mask, ori_shape, rescale=False):
        mask_h, mask_w = mask.shape[-2:]
        ori_h, ori_w = ori_shape
        
        scale = min(mask_h / ori_h, mask_w / ori_w)
        
        new_h = int(ori_h * scale)
        new_w = int(ori_w * scale)
        
        pad_top = (mask_h - new_h) // 2
        pad_left = (mask_w - new_w) // 2
        
        mask = mask[:, pad_top:new_h + pad_top, pad_left:new_w + pad_left]

        if rescale:
            mask = F.interpolate(mask.float().unsqueeze(0), size=ori_shape, 
                                mode="bilinear", align_corners=False).squeeze(0)
        
        return mask


@timeit_evaluator
@memory_evaluator
@EVALUATORS.register(name="AnalysisMMDetDataloaderEvaluator")
class AnalysisMMDetDataloaderEvaluator(COCOEvaluator):
    print('using AnalysisMMDetDataloaderEvaluator')
    # coco_eval
    def __init__(self, cfg: cfg, model=None, dataset=None, **kwargs):
        super().__init__(cfg, model, **kwargs)

        self.dataset = dataset
        outfile_prefix = cfg.model.evaluator.outfile_prefix
        # coco_api = cfg.model.evaluator.get("coco_api", None)
        self.num_classes = cfg.model.decoder.instance_head.num_classes

        print(f"Doing evaluation on dataset.ann_file: {dataset.ann_file}")

        self.metric = CocoMetric(
            ann_file=dataset.ann_file,
            metric=cfg.model.evaluator.metric,
            classwise=cfg.model.evaluator.classwise,
            outfile_prefix=join(cfg.run.save_dir, outfile_prefix) if (outfile_prefix and cfg.run.get("save_dir")) else None,
            # coco_api=coco_api if coco_api else 'COCOeval'
            )

        categories = self.metric._coco_api.loadCats(self.metric._coco_api.getCatIds())
        class_names = [category['name'] for category in categories]
        self.metric.dataset_meta = dict(classes=class_names)

        self.nms_threshold = cfg.model.evaluator.nms_thr


    def forward(self, dataloader):
        super().forward(dataloader)

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), miniters=5)
        for step, batch in pbar:
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

            image = nested_tensor_from_tensor_list(images)


            # ============= PREDICTION ==============
            # predict.
            preds = self.inference_single(image.tensors)
            preds["images"] = image.tensors
            preds["img_id"] = [targets[i]["img_id"] for i in range(len(targets))]
            preds["ori_shape"] = [targets[i]["ori_shape"] for i in range(len(targets))]
            preds["img_path"] = [targets[i]["img_path"] for i in range(len(targets))]

            self.process(preds, step)
    

    def process(self, preds: dict, step=None):
        scores_batch = preds['pred_logits'].softmax(-1)
        masks_pred_batch = preds['pred_instance_masks'].sigmoid()
        iou_scores_batch = preds['pred_scores'].sigmoid()
        bboxes_pred_batch = preds['pred_bboxes']
        
        iams_batch = preds['pred_iams']['instance_iams']
        attn_batch = preds['attn']['mask_pixel_attn']
        # attn_batch = preds['attn']['inst_pixel_attn']
        # attn_batch = preds['attn']['query_sa_attn']
        images_batch = preds['images']

        for batch_idx, (scores, masks_pred, iou_scores, bboxes_pred, iams_pred, attn, image) in enumerate(zip(
            scores_batch, masks_pred_batch, iou_scores_batch, bboxes_pred_batch, iams_batch, attn_batch, images_batch)):
            scores = scores[:, :-1]
            iou_scores = iou_scores.flatten(0, 1)

            labels = torch.arange(self.num_classes, device=scores.device).unsqueeze(0).repeat(masks_pred.shape[0], 1).flatten(0, 1)
            scores, topk_indices = scores.flatten(0, 1).topk(masks_pred.shape[0], sorted=False)
            labels = labels[topk_indices]

            topk_indices = topk_indices // self.num_classes
            masks_pred = masks_pred[topk_indices]
            iou_scores = iou_scores[topk_indices]
            bboxes_pred = bboxes_pred[topk_indices]
            iams_pred = iams_pred[topk_indices]
            # attn = attn[topk_indices]

            # maskness scores.
            seg_masks = masks_pred > self.mask_threshold
            sum_masks = seg_masks.sum((1, 2)).float()
            maskness_scores = (masks_pred * seg_masks.float()).sum((1, 2)) / (sum_masks + 1e-6)
            
            # scores = torch.sqrt(scores * iou_scores)
            scores = scores * maskness_scores

            # ========== CLS Score ==========
            # # score filtering.
            keep = scores > self.score_threshold
            masks_pred = masks_pred[keep]
            scores = scores[keep]
            labels = labels[keep]
            iou_scores = iou_scores[keep]
            bboxes_pred = bboxes_pred[keep]
            iams_pred = iams_pred[keep]
            # attn = attn[keep]

            # ========== NMS ==========
            # pre_nms sort.
            sort_inds = torch.argsort(scores, descending=True)
            masks_pred = masks_pred[sort_inds]
            scores = scores[sort_inds]
            labels = labels[sort_inds]
            iou_scores = iou_scores[sort_inds]
            bboxes_pred = bboxes_pred[sort_inds]
            iams_pred = iams_pred[sort_inds]
            # attn = attn[sort_inds]

            # nms.
            seg_masks = masks_pred > self.mask_threshold
            sum_masks = seg_masks.sum((1, 2)).float()
            
            keep = mask_nms(labels, seg_masks, sum_masks, scores, nms_thr=self.nms_threshold)
            masks_pred = masks_pred[keep]
            scores = scores[keep]
            labels = labels[keep]
            iou_scores = iou_scores[keep]
            bboxes_pred = bboxes_pred[keep]
            iams_pred = iams_pred[keep]
            # attn = attn[keep]

            # postprocessing - currently done here, should be moved to model.
            ori_shape = preds["ori_shape"][batch_idx]
            if masks_pred.shape[0]:
                image = remove_padding(
                    image, 
                    ori_shape,
                    rescale=True
                )
                masks_pred = remove_padding(
                    masks_pred, 
                    ori_shape,
                    rescale=True
                )
                iams_pred = remove_padding(
                    iams_pred, 
                    ori_shape,
                    rescale=True
                )
                attn = remove_padding(
                    attn, 
                    ori_shape,
                    rescale=False
                )

            masks_pred = masks_pred > self.mask_threshold
            # ================================================
            
            # image saving.
            image = image.cpu().numpy()[0, ...]
            masks = masks_pred.cpu().numpy()
            iams = iams_pred.cpu().numpy()
            attn = attn.cpu().numpy()

            # save_image_and_masks(image, masks, iams, 
            #                      img_id=preds["img_id"][batch_idx], 
            #                      step=step,
            #                      shape=ori_shape, 
            #                      dpi=100)
            
            # plot_pred_iam_set(image, masks, iams, 
            #                   img_id=preds["img_id"][batch_idx], 
            #                   step=step,
            #                   shape=ori_shape, 
            #                   dpi=100)

            # plot_attn_set(attn, 
            #               img_id=preds["img_id"][batch_idx], 
            #               step=step, 
            #               shape=ori_shape, 
            #               dpi=100)

            save_attn(attn, 
                      img_id=preds["img_id"][batch_idx], 
                      step=step,
                      shape=ori_shape, 
                      dpi=100)

            # ================================================
            

            results = dict()
            results["img_id"] = preds["img_id"][batch_idx]
            results["ori_shape"] = preds["ori_shape"][batch_idx]
            results["pred_instances"] = {
                "masks": masks_pred,
                "labels": labels,
                "scores": scores,
                "mask_scores": scores,
                "bboxes": bboxes_pred,
            }

            data_samples = [results]
            self.metric.process({}, data_samples)
        

    def evaluate(self, verbose=False):
        key_mapping = {
            'coco/segm_mAP': "mAP@0.5:0.95",
            'coco/segm_mAP_50': "mAP@0.5",
            'coco/segm_mAP_75': "mAP@0.75",
            'coco/segm_mAP_s': "mAP(s)@0.5",
            'coco/segm_mAP_m': "mAP(m)@0.5",
            'coco/segm_mAP_l': "mAP(l)@0.5",
        }

        # Compute metrics
        size = len(self.dataset)
        eval_results = self.metric.evaluate(size)

        # Update self.stats based on the mapping
        for key, value in eval_results.items():
            if key in key_mapping:
                self.stats[key_mapping[key]] = value

        self.gt_coco = self.metric._coco_api
        self.pred_coco = self.metric.coco_dt
