import albumentations as A
import cv2
import random
from configs import cfg

padding = 0

def train_transforms(cfg: cfg):
    _transforms = A.Compose([
#         A.RandomScale(scale_limit=(0, 1.5), p=1),
        
        # A.RandomScale(scale_limit=(-0.5, -0.1), p=1),
        # A.PadIfNeeded(min_height=1024, min_width=1024, value=0, border_mode=0, position='random'),
        
        # A.RandomScale(scale_limit=(-0.5, 0.5), p=1, interpolation=1),
        # A.PadIfNeeded(768, 768, border_mode=cv2.BORDER_CONSTANT),
        # A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT),
            
        # A.OneOf([
        #     A.Compose([
        #         A.Resize(768, 768),
        #         A.RandomScale(scale_limit=(-0.5, 0.5), p=1, interpolation=1),
        #         A.PadIfNeeded(768, 768, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
        #     ]),
        #     # A.RandomCrop(768, 768),
        #     # A.RandomCrop(640, 640),

        #     A.RandomCrop(420, 420),
        #     A.RandomCrop(384, 384),

        #     # A.Compose([
        #     #     A.RandomScale(scale_limit=(-0.5, 0.5), p=1, interpolation=1),
        #     #     A.PadIfNeeded(768, 768, border_mode=cv2.BORDER_CONSTANT),
        #     #     A.RandomCrop(768, 768),
        #     # ]),
        #     # A.Compose([
        #     #     A.RandomScale(scale_limit=(-0.2, 0.2), p=1, interpolation=1),
        #     #     A.PadIfNeeded(640, 640, border_mode=cv2.BORDER_CONSTANT),
        #     #     A.RandomCrop(640, 640),
        #     # ]),
        # ], p=0.5),

        # A.RandomCrop(768, 768),
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

        # A.OneOf([
        #     A.RandomCrop(512, 512)
        # ], p=0.5),
        # A.CoarseDropout(max_holes=2, max_height=200, max_width=200, min_holes=None, min_height=None, min_width=None, 
        #                 fill_value=0, mask_fill_value=None, always_apply=False, p=0.5),

        # # A.RandomCrop(768, 768),
        # A.Resize(512, 512),  

        A.ElasticTransform(
            alpha=10,  # Adjust alpha to control deformation strength
            sigma=10,  # Adjust sigma to control the spatial smoothness
            alpha_affine=10,
            interpolation=1,
            border_mode=cv2.BORDER_CONSTANT,
            value=None,
            mask_value=None,
            always_apply=False,
            approximate=True,  # Use approximate elastic transform for speed
            same_dxdy=False,
            p=1
        ),
        
#         A.ShiftScaleRotate(shift_limit=0, scale_limit=(-1, -0.5), rotate_limit=0, p=1),
        
        # A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, 
        #                    border_mode=cv2.BORDER_CONSTANT, value=None, mask_value=None, 
        #                    always_apply=False, approximate=False, same_dxdy=False, p=0.5),
        
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



# def train_transforms(cfg):
#     return A.Compose([
#         A.OneOf([
#             A.Compose([
#                 A.RandomScale(scale_limit=(-0.5, 0.5), p=1, interpolation=1),
#                 A.PadIfNeeded(768, 768, border_mode=cv2.BORDER_CONSTANT),
#             ]),
#             A.Compose([
#                 A.RandomScale(scale_limit=(-0.5, 0.5), p=1, interpolation=1),
#                 A.PadIfNeeded(768, 768, border_mode=cv2.BORDER_CONSTANT),
#                 A.RandomCrop(768, 768),
#             ]),
#             A.Compose([
#                 A.RandomScale(scale_limit=(-0.2, 0.2), p=1, interpolation=1),
#                 A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT),
#                 A.RandomCrop(512, 512),
#             ]),
#             # A.Compose([
#             #     A.RandomScale(scale_limit=(-0.1, 0.1), p=1, interpolation=1),
#             #     A.PadIfNeeded(384, 384, border_mode=cv2.BORDER_CONSTANT),
#             #     A.RandomCrop(384, 384),
#             # ]),
#         ], p=1),
#         A.Resize(*cfg.train.size),

#         A.VerticalFlip(p=0.5),
#         A.HorizontalFlip(p=0.5),
#         A.RandomRotate90(p=1),

#         A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, 
#                            border_mode=cv2.BORDER_CONSTANT, value=None, mask_value=None, 
#                            always_apply=False, approximate=False, same_dxdy=False, p=0.5),
                           
#     ], additional_targets={'prob_map': 'mask'})


# def train_transforms(cfg):
#     scales = [(480, 480), (512, 512), (544, 544), (576, 576), (608, 608),
#               (640, 640), (672, 672), (704, 704), (736, 736), (768, 768),
#               (800, 800)]
#     small_scales = [(400, 400), (500, 500), (600, 600)]
#     max_long_side = 1333
#     crop_size = (384, 384)
    
#     return A.OneOf([
#         # First Branch: Direct resize
#         A.Sequential([
#             A.Resize(height=random.choice(scales)[0], width=random.choice(scales)[1], always_apply=True),
#         ], p=0.5),

#         # Second Branch: Crop and then resize
#         A.Sequential([
#             A.Resize(height=random.choice(small_scales)[0], width=random.choice(small_scales)[1], always_apply=True),
#             A.RandomCrop(height=crop_size[0], width=crop_size[1], always_apply=True),
#             A.Resize(height=random.choice(scales)[0], width=random.choice(scales)[1], always_apply=True),
#         ], p=0.5),
#     ], p=1)




# def valid_transforms(cfg: cfg):
#     _transforms = A.Compose([
# #         A.RandomCrop(768, 768),
#         A.Resize(*cfg.valid.size),
        
#     # ], additional_targets={'image1': 'image', 'mask1': 'mask', 'overlap': 'mask', 'dist_map': 'mask'})
#     ], additional_targets={'prob_map': 'mask'})

#     return _transforms




# EVICAN
# def train_transforms(cfg):
#     return A.Compose([
#         # A.OneOf([
#         #     A.Compose([
#         #         A.RandomScale(scale_limit=(-0.5, 0.5), p=1, interpolation=1),
#         #         A.PadIfNeeded(768, 768, border_mode=cv2.BORDER_CONSTANT),
#         #         A.RandomCrop(768, 768),
#         #     ]),
#         #     A.Compose([
#         #         A.RandomScale(scale_limit=(-0.2, 0.2), p=1, interpolation=1),
#         #         A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT),
#         #         A.RandomCrop(512, 512),
#         #     ]),
#         # ], p=0.5),

#         # A.RandomScale(scale_limit=(-0.5, 0.5), p=1, interpolation=1),
#         # A.PadIfNeeded(768, 768, border_mode=cv2.BORDER_CONSTANT),

#         A.Compose([
#             A.Resize(768, 768),
#             A.RandomScale(scale_limit=(-0.5, 0.5), p=1, interpolation=1),
#             A.PadIfNeeded(768, 768, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
#         ], p=0.5),
            
#         # A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT),
#         # A.CropNonEmptyMaskIfExists(512, 512),
#         # A.OneOf([
#         #     A.Resize(512, 512),
#         #     A.RandomCrop(512, 512),
#         # ], p=0.5),

#         A.Resize(512, 512),

#         A.VerticalFlip(p=0.5),
#         A.HorizontalFlip(p=0.5),
#         A.RandomRotate90(p=1),

#         A.ElasticTransform(
#             alpha=10,  # Adjust alpha to control deformation strength
#             sigma=10,  # Adjust sigma to control the spatial smoothness
#             alpha_affine=50,
#             interpolation=1,
#             border_mode=cv2.BORDER_CONSTANT,
#             value=None,
#             mask_value=None,
#             always_apply=False,
#             approximate=True,  # Use approximate elastic transform for speed
#             same_dxdy=False,
#             p=1
#         ),
#     ])

# def valid_transforms(cfg):
#     _transforms = A.Compose([
#         # A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT),
#         # A.Crop(0, 0, 256, 256), #.RandomCrop(150, 150),
#         A.Resize(512, 512),
#     ])
#     return _transforms



# LiveCell
# def train_transforms(cfg):
#     _transforms = A.Compose([
#         A.RandomCrop(150, 150),
#         # A.RandomCrop(256, 256),

#         # A.RandomScale(scale_limit=(-0.5, 0.5), p=1, interpolation=1),
#         # A.PadIfNeeded(256, 256, border_mode=cv2.BORDER_CONSTANT),

#         # A.Compose([
#         #     A.Resize(512, 512),
#         #     A.RandomScale(scale_limit=(-0.5, 0.5), p=1, interpolation=1),
#         #     A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
#         # ], p=1),

#         A.Resize(512, 512),

#         A.VerticalFlip(p=0.5),
#         A.HorizontalFlip(p=0.5),
#         A.RandomRotate90(p=1),

#         # A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, 
#         #                     border_mode=cv2.BORDER_CONSTANT, value=None, mask_value=None, 
#         #                     always_apply=False, approximate=False, same_dxdy=False, p=0.5),

#     ])
#     return _transforms

# def valid_transforms(cfg):
#     _transforms = A.Compose([
#         A.Crop(0, 0, 512, 512), #.RandomCrop(150, 150),
#         # A.Crop(0, 0, 256, 256),
#         # A.Resize(512, 512),
#     ])

#     return _transforms