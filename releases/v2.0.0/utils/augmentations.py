import albumentations as A
import cv2
from configs import cfg

padding = 0

def train_transforms(cfg: cfg):
    _transforms = A.Compose([
#         A.RandomScale(scale_limit=(0, 1.5), p=1),
        
        # A.RandomScale(scale_limit=(-0.5, -0.1), p=1),
        # A.PadIfNeeded(min_height=1024, min_width=1024, value=0, border_mode=0, position='random'),
        
        A.RandomScale(scale_limit=(-0.5, 0.5), p=1, interpolation=1),
        A.PadIfNeeded(768, 768, border_mode=cv2.BORDER_CONSTANT),
        
#         A.RandomCrop(768, 768),
        A.Resize(*cfg.train.size),
        
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=1),

        # faster multi-scale transforms
        # A.Resize(512, 512), 
        # A.VerticalFlip(p=0.5),
        # A.HorizontalFlip(p=0.5),
        # A.RandomRotate90(p=1),
        # A.RandomScale(scale_limit=(-0.2, 0.2), p=1, interpolation=1), 
        # A.PadIfNeeded(768, 768, border_mode=cv2.BORDER_CONSTANT), 
        # # A.RandomCrop(768, 768),
        # A.Resize(512, 512),  

        # A.ElasticTransform(
        #     alpha=10,  # Adjust alpha to control deformation strength
        #     sigma=10,  # Adjust sigma to control the spatial smoothness
        #     alpha_affine=10,
        #     interpolation=1,
        #     border_mode=cv2.BORDER_CONSTANT,
        #     value=None,
        #     mask_value=None,
        #     always_apply=False,
        #     approximate=True,  # Use approximate elastic transform for speed
        #     same_dxdy=False,
        #     p=1
        # ),
        
#         A.ShiftScaleRotate(shift_limit=0, scale_limit=(-1, -0.5), rotate_limit=0, p=1),
        
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, 
                           border_mode=cv2.BORDER_CONSTANT, value=None, mask_value=None, 
                           always_apply=False, approximate=False, same_dxdy=False, p=0.5),
        
#         A.CoarseDropout(max_holes=6, min_holes=2, max_height=50, max_width=50, 
#                         min_height=25, min_width=25, mask_fill_value=0, p=0.5),
    # ], additional_targets={'image1': 'image', 'mask1': 'mask', 'overlap': 'mask', 'dist_map': 'mask'})
    ], additional_targets={'prob_map': 'mask'})

    return _transforms


def valid_transforms(cfg: cfg):
    _transforms = A.Compose([
#         A.RandomCrop(768, 768),
        A.Resize(*cfg.valid.size),
        
    # ], additional_targets={'image1': 'image', 'mask1': 'mask', 'overlap': 'mask', 'dist_map': 'mask'})
    ], additional_targets={'prob_map': 'mask'})

    return _transforms