import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform

class Resize(A.DualTransform):
    """Resize the image to a specified minimum short side and maximum long side.
    
    Args:
        scale (tuple): Desired size as (min_short_side, max_long_side).
        interpolation (int): Interpolation method for resizing.
            Default is cv2.INTER_LINEAR.
        always_apply (bool): Whether to always apply this transform.
            Default is True.
        p (float): Probability of applying this transform.
            Default is 1.0.
    """
    def __init__(
        self, 
        scale: tuple = (800, 1333),
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = True,
        p: float = 1.0
    ):
        super().__init__(always_apply, p)
        self.min_short_side = scale[0]
        self.max_long_side = scale[1]
        self.interpolation = interpolation

    def apply(self, img, scale, **params):
        h, w = img.shape[:2]
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        return cv2.resize(img, (new_w, new_h), interpolation=self.interpolation)

    def get_params_dependent_on_data(self, params, data):
        h, w = params["shape"][:2]

        scale = self.min_short_side / min(h, w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))

        if max(new_h, new_w) > self.max_long_side:
            scale2 = self.max_long_side / max(new_h, new_w)
            scale = scale * scale2
        return {"scale": scale}
    
    def apply_to_bboxes(self, bboxes, **params):
        # Bounding box coordinates are scale invariant
        return bboxes

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("min_short_side", "max_long_side", "interpolation")