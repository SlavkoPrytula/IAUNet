import numpy as np
from os.path import join
import cv2
from PIL import Image

import sys
sys.path.append('.')

from utils.utils import flatten_mask
from configs import cfg

from utils.registry import DATASETS
from dataset.datasets.base_coco_dataset import BaseCOCODataset



@DATASETS.register(name="YeastNet")
class YeastNet(BaseCOCODataset):
    def __init__(self, cfg: cfg, dataset_type="train", normalization=None, transform=None):
        super().__init__(cfg, dataset_type, normalization, transform)
    
    def get_image(self, img_id):
        img_id = self.image_ids[img_id]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = join(self.img_folder, img_info['file_name'])

        image = Image.open(img_path)
        image = np.array(image, dtype=np.float32)
        image = np.repeat(image[..., np.newaxis], 3, axis=-1)

        if image is None:
            raise FileNotFoundError(f"Image with id {img_id} not found in path: {img_path}")

        return image


if __name__ == "__main__":
    from utils.visualise import visualize, visualize_grid_v2, plot3d
    from utils.normalize import normalize
    from utils.augmentations import train_transforms, valid_transforms
    from utils.registry import DATASETS_CFG
    import time

    time_s = time.time()
    
    cfg.dataset = DATASETS_CFG.get("YeastNet")

    time_s = time.time()
    dataset = YeastNet(cfg, dataset_type="train",
                      normalization=normalize,
                      transform=train_transforms(cfg)
                      )
    
    print(len(dataset))
    
    targets = dataset[4]
    time_e = time.time()
    print(f'loaded in {time_e - time_s}(s)')

    print(targets["image"].shape, targets["ori_shape"])
    print(targets["labels"])

    print(targets["image"].shape, targets["masks"].shape)
    print(f'std: {targets["image"].std()}, mean: {targets["image"].mean()}')

    
    visualize(images=targets["image"][0, ...], path='./test_image.jpg', cmap='gray',)
    visualize(images=flatten_mask(targets["masks"].numpy(), axis=0)[0], path='./test_mask.jpg', cmap='gray',)
    