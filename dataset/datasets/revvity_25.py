import sys
sys.path.append('.')

import hydra
from configs import cfg

from dataset.datasets.base_coco_dataset import BaseCOCODataset
from utils.registry import DATASETS


@DATASETS.register(name="Revvity_25")
class Revvity_25(BaseCOCODataset):
    def __init__(self, cfg: cfg, dataset_type="train", normalization=None, transform=None):
        super().__init__(cfg, dataset_type, normalization, transform)
        

@hydra.main(version_base="1.3", config_path="../../configs", config_name="train")
def main(cfg: cfg):
    from utils.visualise import visualize, visualize_grid_v2
    from visualizations import visualize_masks
    from utils.augmentations import train_transforms, valid_transforms
    import time

    time_s = time.time()

    dataset = Revvity_25(cfg, dataset_type="train", 
                         transform=valid_transforms(cfg)
                         )
    
    print(len(dataset))
    
    targets = dataset[27]
    time_e = time.time()
    print(f'loaded in {time_e - time_s}(s)')

    print(targets["image"].shape, targets["ori_shape"])
    print(targets["labels"])
    print(len(targets["labels"]), len(targets["instance_masks"]))

    print(targets["image"].shape, targets["instance_masks"].shape)
    print(f'std: {targets["image"].std()}, mean: {targets["image"].mean()}')

    visualize(
        images=targets["image"][0, ...], 
        path='./test_image.jpg', 
        cmap='gray',
        show_title=False
    )

    H, W = targets["image"].shape[-2:] #targets["ori_shape"]
    # H, W = targets["ori_shape"]
    visualize_masks(
        figsize=[30, 30],
        img=targets["image"][0, ...],
        masks=targets["instance_masks"],
        shape=[H, W],
        alpha=0.65,
        draw_border=True, 
        static_color=False,
        path='./test_mask.png',
        dpi=300
        # show_img=True
    )

    
    # import torch.nn.functional as F
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from matplotlib import cm
    # from scipy.ndimage import binary_dilation, binary_erosion

    # img = targets["image"][0, ...]  # (H_orig, W_orig)
    # masks = targets["instance_masks"]  # (N, H_orig, W_orig)
    # shape = (H, W)
    # border_size = 1
    # alpha = 0.75  # transparency for the mask

    # # Resize image and masks
    # img_resized = F.interpolate(img.unsqueeze(0).unsqueeze(0), size=shape,
    #                             mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
    # masks_resized = F.interpolate(masks.unsqueeze(0).float(), size=shape,
    #                             mode="bilinear", align_corners=False).squeeze(0)

    # # Merge instance masks
    # binary_mask = masks_resized.sum(dim=0).clamp(0, 1) > 0  # (H, W) binary

    # # Convert grayscale image to RGB
    # img_np = img_resized.cpu().numpy()
    # img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-5)
    # img_rgb = np.stack([img_np]*3, axis=-1)  # (H, W, 3)

    # # === Create RGBA mask ===
    # rgba_mask = np.zeros((*binary_mask.shape, 4), dtype=np.float32)

    # # Yellow color from viridis
    # yellow_rgb = cm.get_cmap("viridis")(0.95)[:3]

    # # Set RGB and alpha where mask is present
    # for c in range(3):
    #     rgba_mask[..., c][binary_mask] = yellow_rgb[c]
    # rgba_mask[..., 3][binary_mask] = alpha  # Alpha channel

    # # === Add white border to RGBA mask ===
    # binary_mask_for_border = rgba_mask[..., 3] > 0
    # dilation = binary_dilation(binary_mask_for_border, iterations=border_size + 2)
    # erosion = binary_erosion(binary_mask_for_border, iterations=border_size)
    # border = dilation & ~erosion

    # for c in range(3):
    #     rgba_mask[..., c][border] = 1.0  # White border
    # rgba_mask[..., 3][border] = 1.0      # Fully opaque border

    # # === Show result ===
    # plt.figure(figsize=(30, 30))
    # plt.imshow(img_rgb, cmap='gray')
    # plt.imshow(rgba_mask) 
    # plt.axis("off")
    # plt.savefig('./test_binary.png', bbox_inches='tight', pad_inches=0, dpi=100)

    # visualize_grid_v2(
    #     masks=targets["instance_masks"].numpy(), 
    #     path='./test_inst.jpg',
    #     ncols=5
    # )

if __name__ == "__main__":
    main()
