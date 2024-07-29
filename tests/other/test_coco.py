import sys
sys.path.append("./")

from utils.coco.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.coco.mask2coco import masks2coco

import numpy as np
import matplotlib.pyplot as plt


def flatten_mask(mask, axis=-1):
    f = np.sum(mask, axis)
    f = np.expand_dims(f, axis)
    return f


# create mask 1 of shape (N, H, W)
mask_1 = np.zeros((100, 512, 512))
# mask_1[0, :5, :5] = 1
# mask_1[1, 7:10, 8:9] = 1
# mask_1[2, 9:10, 9:10] = 1
# mask_1[3, :4, 9:10] = 1

for i in range(100):
    x = np.random.randint(0, 256)
    y = np.random.randint(0, 256)
    w = np.random.randint(1, 40)
    h = np.random.randint(1, 40)
    
    mask_1[i, y:y+h, x:x+w] = 1

    


# create mask 2 of shape (N, H, W)
# mask_2 = np.zeros((200, 10, 10))
# mask_2[0, :5, :5] = 1
# mask_2[1, 7:10, 8:9] = 1
# mask_2[2, 9:10, 9:10] = 1
# mask_2[3, :4, 9:10] = 1
# mask_2[4, :5, 9:10] = 1

mask_2 = mask_1.copy()

# prepare masks
gt_masks = [mask_1]
pred_masks = [mask_2]

import time
start_time = time.time()

# masks -> coco
gt_coco = masks2coco(gt_masks)
pred_coco = masks2coco(pred_masks)
# pred_coco = masks2coco(pred_masks, scores=[[0.1, 0.5, 0.5, 0.5, 1]])

end_time = time.time()

# print(pred_coco)
# print(len(pred_coco["images"]))

# load coco dict
gt_coco = COCO(gt_coco)
pred_coco = COCO(pred_coco)
coco_eval = COCOeval(gt_coco, pred_coco, iouType='segm')
coco_eval.params.maxDets = [200, 200, 200]

# Run the evaluation
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# plot
mask_1 = flatten_mask(mask_1, axis=0)   # overlay masks: (N, H, W) -> (1, H, W)
mask_2 = flatten_mask(mask_2, axis=0)   # overlay masks: (N, H, W) -> (1, H, W)

fig, ax = plt.subplots(1, 2, figsize=[10, 5])
ax[0].imshow(mask_1[0])
ax[1].imshow(mask_2[0])
plt.savefig("./test_coco.jpg")

print(end_time - start_time)