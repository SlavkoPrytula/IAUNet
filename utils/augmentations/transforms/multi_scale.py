import albumentations as A
import cv2
import random


class RandomChoiceResize(A.DualTransform):
    """Resize the shortest side of the image to a random size from a given list.

    Args:
        sizes (list of int): Possible sizes to select from for the shortest side.
        interpolation (cv2 InterpolationFlags): Desired interpolation enum from OpenCV.
            Defaults to cv2.INTER_LINEAR.
        always_apply (bool): Indicates whether this transformation should always be applied.
        p (float): Probability of applying the transform. Default: 1.
    """

    def __init__(self, sizes, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1.0):
        super(RandomChoiceResize, self).__init__(always_apply, p)
        self.sizes = sizes
        self.interpolation = interpolation

    def apply(self, img, **params):
        self.size = random.choice(self.sizes)
        resize_transform = A.SmallestMaxSize(max_size=self.size, interpolation=self.interpolation)
        img = resize_transform.apply(img, max_size=self.size)
        return img

    def get_transform_init_args_names(self):
        return ('sizes', 'interpolation', 'mask_interpolation')
