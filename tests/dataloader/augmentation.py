import cv2
import numpy as np
import albumentations as A
from matplotlib import pyplot as plt


def train_transforms():
    f = np.random.uniform(0.75, 1)
    print(f)
    _transforms = A.Compose([
         A.OneOf([
            A.RandomResizedCrop(height=512, 
                                width=512, 
                                scale=(0.5, 1.0),
                                ratio=(0.75, 1.33),
                                interpolation=cv2.INTER_LINEAR, 
                                p=1),
            A.Compose([
                A.Resize(int(512 * f), int(512 * f)),
                A.PadIfNeeded(min_height=512, 
                              min_width=512, 
                              border_mode=cv2.BORDER_CONSTANT, 
                              value=0)
            ])
        ], p=1.0),
        A.OneOf([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=1),
        ], p=1.0),
    ])
    return _transforms

image = np.zeros((512, 512, 3))
image[:100, :100] = 1
image[100:150, 100:150] = 1

transform = train_transforms()
augmented = transform(image=image)['image']
print(augmented.shape)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(augmented)
axes[1].set_title('Augmented Image')
axes[1].axis('off')

plt.savefig("temp.png")




# time test augmentations.

# import cv2
# import numpy as np
# import albumentations as A
# import time
# from tqdm import tqdm

# def train_transforms_v1(cfg):
#     _transforms = A.Compose([
#         A.Resize(*cfg['size']),
#         A.RandomScale(scale_limit=(-0.2, 0.5), p=1, interpolation=cv2.INTER_LINEAR),
#         A.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT),
#         A.RandomCrop(512, 512),
#         A.VerticalFlip(p=0.5),
#         A.HorizontalFlip(p=0.5),
#         A.RandomRotate90(p=1),
#     ])
#     return _transforms

# def train_transforms_v2(cfg):
#     f = np.random.uniform(0.7, 0.9)
#     _transforms = A.Compose([
#         A.OneOf([
#             A.RandomResizedCrop(height=512, 
#                                 width=512, 
#                                 scale=(0.5, 1.0),
#                                 ratio=(0.75, 1.33),
#                                 interpolation=cv2.INTER_LINEAR, 
#                                 p=1),
#             A.Compose([
#                 A.Resize(int(512 * f), int(512 * f)),
#                 A.PadIfNeeded(min_height=512, 
#                               min_width=512, 
#                               border_mode=cv2.BORDER_CONSTANT, 
#                               value=0)
#             ])
#         ], p=1.0),

#         A.OneOf([
#             A.VerticalFlip(p=0.5),
#             A.HorizontalFlip(p=0.5),
#             A.RandomRotate90(p=1),
#         ], p=1.0),
#     ])
#     return _transforms

# cfg = {'size': [512, 512]}
# image = np.random.rand(512, 512, 3).astype(np.float32)


# transform_v1 = train_transforms_v1(cfg)
# start_time_v1 = time.time()
# for _ in tqdm(range(5000)):
#     augmented_v1 = transform_v1(image=image)['image']
# end_time_v1 = time.time()

# transform_v2 = train_transforms_v2(cfg)
# start_time_v2 = time.time()
# for _ in tqdm(range(5000)):
#     augmented_v2 = transform_v2(image=image)['image']
# end_time_v2 = time.time()


# time_v1 = end_time_v1 - start_time_v1
# time_v2 = end_time_v2 - start_time_v2
# print(time_v1)
# print(time_v2)
