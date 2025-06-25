import random
import numpy as np
import albumentations as A
from skimage.filters import gaussian

def image_copy_paste(img, paste_img, alpha, blend=True, sigma=1):
    if alpha is not None:
        if blend:
            alpha = gaussian(alpha, sigma=sigma, preserve_range=True)

        img_dtype = img.dtype
        alpha = alpha[..., None]
        img = paste_img * alpha + img * (1 - alpha)
        img = img.astype(img_dtype)

    return img

def mask_copy_paste(mask, paste_mask, alpha):
    raise NotImplementedError

def masks_copy_paste(masks, paste_masks, alpha):
    """Copy-paste masks with alpha blending."""
    if alpha is None:
        return masks
    
    # Convert masks to numpy array format for easier manipulation
    if isinstance(masks, np.ndarray):
        if masks.ndim == 3:  # (H, W, C) format
            mask_array = masks.copy()
        else:  # (H, W) format - single mask
            mask_array = masks[..., np.newaxis].copy()
    else:
        # Convert list to array
        if len(masks) > 0:
            mask_array = np.stack(masks, axis=-1)
        else:
            return masks
    
    # Apply alpha mask to existing masks (remove overlapping areas)
    # Only remove significant overlaps to preserve original objects
    alpha_3d = alpha[..., np.newaxis]  # Broadcast alpha to all channels
    overlap_threshold = 0.5  # Only remove areas with significant overlap
    
    # Create a softer removal - only remove where alpha is strong
    removal_mask = (alpha_3d > overlap_threshold).astype(np.float32)
    mask_array = mask_array * (1 - removal_mask)  # Zero out areas with significant overlap
    
    # Add paste masks (these are the shifted versions)
    if paste_masks is not None:
        for paste_mask in paste_masks:
            # Create a new channel for the pasted mask
            paste_mask_3d = paste_mask[..., np.newaxis]
            mask_array = np.concatenate([mask_array, paste_mask_3d], axis=-1)
    
    return mask_array

def extract_bboxes(masks):
    """Extract bounding boxes from masks in Pascal VOC format (x1, y1, x2, y2)."""
    bboxes = []
    if len(masks) == 0:
        return bboxes
    
    for mask in masks:
        if isinstance(mask, np.ndarray):
            if mask.ndim == 3:
                mask = mask.squeeze()
            
            # Find non-zero pixels
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            if rows.any() and cols.any():
                y_indices = np.where(rows)[0]
                x_indices = np.where(cols)[0]
                
                x1, x2 = x_indices[[0, -1]]
                y1, y2 = y_indices[[0, -1]]
                
                # Convert to Pascal VOC format (x1, y1, x2, y2)
                bboxes.append([x1, y1, x2 + 1, y2 + 1])
            else:
                # Empty mask
                bboxes.append([0, 0, 0, 0])
        else:
            bboxes.append([0, 0, 0, 0])

    return bboxes

def bboxes_copy_paste(bboxes, paste_bboxes, masks, paste_masks, alpha, key):
    """Handle bbox updates after copy-paste operation."""
    if alpha is None:
        return bboxes
    
    # Convert masks to list format if needed
    if masks is not None:
        if isinstance(masks, np.ndarray):
            if masks.ndim == 3:  # (H, W, C) format
                masks_list = [masks[..., i] for i in range(masks.shape[-1])]
            else:  # (H, W) format - single mask
                masks_list = [masks]
        else:
            masks_list = masks
    else:
        masks_list = []
    
    # Apply alpha mask to existing masks and extract new bboxes
    updated_bboxes = []
    for i, mask in enumerate(masks_list):
        # Remove areas that will be pasted over
        updated_mask = np.logical_and(mask, np.logical_not(alpha)).astype(np.uint8)
        
        # Extract bbox from updated mask
        rows = np.any(updated_mask, axis=1)
        cols = np.any(updated_mask, axis=0)
        
        if rows.any() and cols.any():
            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            
            x1, x2 = x_indices[[0, -1]]
            y1, y2 = y_indices[[0, -1]]
            
            # Create new bbox, preserving labels from original
            new_bbox = [x1, y1, x2 + 1, y2 + 1]
            if i < len(bboxes) and len(bboxes[i]) > 4:
                new_bbox.extend(bboxes[i][4:])  # Preserve labels
            elif i < len(bboxes):
                # If original bbox doesn't have extra fields, preserve what we can
                new_bbox.append(bboxes[i][-1] if len(bboxes[i]) > 4 else 1)  # Default label
            updated_bboxes.append(new_bbox)
    
    # Add bboxes from paste masks
    if paste_masks is not None:
        if isinstance(paste_masks, np.ndarray):
            if paste_masks.ndim == 3:  # (H, W, C) format
                paste_masks_list = [paste_masks[..., i] for i in range(paste_masks.shape[-1])]
            else:  # (H, W) format - single mask
                paste_masks_list = [paste_masks]
        else:
            paste_masks_list = paste_masks
        
        paste_bboxes_extracted = extract_bboxes(paste_masks_list)
        for i, paste_bbox in enumerate(paste_bboxes_extracted):
            if paste_bbox != [0, 0, 0, 0]:  # Non-empty bbox
                # Add a default label for pasted objects
                paste_bbox.append(2)  # Default label for pasted objects
                updated_bboxes.append(paste_bbox)
    
    return updated_bboxes

def keypoints_copy_paste(keypoints, paste_keypoints, alpha):
    #remove occluded keypoints
    if alpha is not None:
        visible_keypoints = []
        for kp in keypoints:
            x, y = kp[:2]
            tail = kp[2:]
            if alpha[int(y), int(x)] == 0:
                visible_keypoints.append(kp)

        keypoints = visible_keypoints + paste_keypoints

    return keypoints

class CopyPaste(A.DualTransform):
    """Copy-Paste augmentation that works out of the box with standard Albumentations data.
    
    This implementation performs copy-paste by randomly selecting objects from the current
    image/masks and duplicating them to different locations within the same image.
    """
    
    def __init__(
        self,
        blend=True,
        sigma=3,
        pct_objects_paste=0.3,
        max_paste_objects=None,
        p=0.5,
        always_apply=False
    ):
        super(CopyPaste, self).__init__(always_apply, p)
        self.blend = blend
        self.sigma = sigma
        self.pct_objects_paste = pct_objects_paste
        self.max_paste_objects = max_paste_objects

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask, "bboxes": self.apply_to_bboxes, "labels": self.apply_to_labels}

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_params_dependent_on_targets(self, params):
        """Generate parameters for copy-paste augmentation."""
        masks = params.get("mask", None)
        
        # Handle different mask formats
        if masks is None:
            return self._get_empty_params()
        
        # Convert masks to list format
        if isinstance(masks, np.ndarray):
            if masks.ndim == 3:  # (H, W, C) format
                masks_list = [masks[..., i] for i in range(masks.shape[-1])]
            elif masks.ndim == 2:  # (H, W) format - single mask
                masks_list = [masks]
            else:
                return self._get_empty_params()
        else:
            masks_list = masks
        
        if len(masks_list) == 0:
            return self._get_empty_params()
        
        # Number of objects available
        n_objects = len(masks_list)
        
        # Calculate number of objects to paste
        n_select = max(1, int(n_objects * self.pct_objects_paste))
        if self.max_paste_objects:
            # n_select = min(n_select, self.max_paste_objects)
            n_select = np.random.randint(1, self.max_paste_objects)
        
        # For copy-paste to be meaningful, we need at least one object
        if n_objects < 1:
            return self._get_empty_params()
        
        # Select random objects to paste
        available_indices = list(range(n_objects))
        # n_select = min(n_select, n_objects)
        if n_select <= 0:
            return self._get_empty_params()
            
        objs_to_paste = np.random.choice(
            available_indices, size=n_select, replace=True
        )
        
        # Get the selected masks
        selected_masks = [masks_list[idx] for idx in objs_to_paste]
        
        # Generate separate shifts for each object to avoid overlap
        h, w = selected_masks[0].shape
        max_shift_x = w // 3  # Allow shifting up to 1/3 of width
        max_shift_y = h // 3  # Allow shifting up to 1/3 of height
        min_shift = min(w, h) // 8  # Minimum shift to make copy-paste visible
        
        # Store individual object info
        paste_objects = []
        combined_alpha = np.zeros((h, w), dtype=np.float32)
        
        for i, mask in enumerate(selected_masks):
            # Generate unique shift for each object
            shift_x = np.random.randint(-max_shift_x, max_shift_x + 1)
            shift_y = np.random.randint(-max_shift_y, max_shift_y + 1)
            
            # If shift is too small, make it bigger
            if abs(shift_x) < min_shift and abs(shift_y) < min_shift:
                if np.random.random() > 0.5:
                    shift_x = min_shift if np.random.random() > 0.5 else -min_shift
                else:
                    shift_y = min_shift if np.random.random() > 0.5 else -min_shift
            
            # Create alpha mask for this object
            object_alpha = (mask > 0).astype(np.float32)
            
            # Apply shift to this object's alpha mask
            alpha_shifted = np.zeros_like(object_alpha)
            
            if shift_x != 0 or shift_y != 0:
                # Calculate source and destination bounds
                src_y_start = max(0, -shift_y)
                src_y_end = min(h, h - shift_y)
                src_x_start = max(0, -shift_x)
                src_x_end = min(w, w - shift_x)
                
                dst_y_start = max(0, shift_y)
                dst_y_end = dst_y_start + (src_y_end - src_y_start)
                dst_x_start = max(0, shift_x)
                dst_x_end = dst_x_start + (src_x_end - src_x_start)
                
                # Copy the shifted region
                alpha_shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                    object_alpha[src_y_start:src_y_end, src_x_start:src_x_end]
            else:
                alpha_shifted = object_alpha.copy()
            
            # Store object info
            paste_objects.append({
                'mask': mask,
                'alpha': alpha_shifted,
                'shift_x': shift_x,
                'shift_y': shift_y
            })
            
            # Add to combined alpha (for removing from original masks)
            combined_alpha = np.logical_or(combined_alpha, alpha_shifted).astype(np.float32)
        
        return {
            "alpha": combined_alpha,
            "paste_objects": paste_objects,
            "objs_to_paste": objs_to_paste
        }
    
    def _get_empty_params(self):
        """Return empty parameters when no copy-paste should be applied."""
        return {
            "alpha": None,
            "paste_objects": [],
            "objs_to_paste": []
        }

    def apply(self, img, alpha, paste_objects, **params):
        """Apply copy-paste to image."""
        if alpha is None or len(paste_objects) == 0:
            return img
        
        result_img = img.copy()
        h, w = img.shape[:2]
        
        # Apply each paste object separately
        for paste_obj in paste_objects:
            object_alpha = paste_obj['alpha']
            shift_x = paste_obj['shift_x']
            shift_y = paste_obj['shift_y']
            
            # Create shifted version of the image for this object
            paste_img = np.zeros_like(img)
            
            if shift_x != 0 or shift_y != 0:
                # Calculate source and destination bounds
                src_y_start = max(0, -shift_y)
                src_y_end = min(h, h - shift_y)
                src_x_start = max(0, -shift_x)
                src_x_end = min(w, w - shift_x)
                
                dst_y_start = max(0, shift_y)
                dst_y_end = dst_y_start + (src_y_end - src_y_start)
                dst_x_start = max(0, shift_x)
                dst_x_end = dst_x_start + (src_x_end - src_x_start)
                
                # Copy the shifted region
                paste_img[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                    img[src_y_start:src_y_end, src_x_start:src_x_end]
            else:
                paste_img = img.copy()
            
            # Apply this object's paste
            result_img = image_copy_paste(
                result_img, paste_img, object_alpha, blend=self.blend, sigma=self.sigma
            )
        
        return result_img

    def apply_to_mask(self, masks, alpha, paste_objects, **params):
        """Apply copy-paste to masks."""
        if alpha is None or len(paste_objects) == 0:
            return masks
        
        # Create shifted paste masks for each object
        shifted_paste_masks = []
        for paste_obj in paste_objects:
            paste_mask = paste_obj['mask']
            shift_x = paste_obj['shift_x']
            shift_y = paste_obj['shift_y']
            
            h, w = paste_mask.shape
            shifted_mask = np.zeros_like(paste_mask)
            
            if shift_x != 0 or shift_y != 0:
                # Calculate source and destination bounds
                src_y_start = max(0, -shift_y)
                src_y_end = min(h, h - shift_y)
                src_x_start = max(0, -shift_x)
                src_x_end = min(w, w - shift_x)
                
                dst_y_start = max(0, shift_y)
                dst_y_end = dst_y_start + (src_y_end - src_y_start)
                dst_x_start = max(0, shift_x)
                dst_x_end = dst_x_start + (src_x_end - src_x_start)
                
                # Copy the shifted region
                shifted_mask[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                    paste_mask[src_y_start:src_y_end, src_x_start:src_x_end]
            else:
                shifted_mask = paste_mask.copy()
            
            shifted_paste_masks.append(shifted_mask)
        
        result_masks = masks_copy_paste(masks, shifted_paste_masks, alpha)
        
        # Convert back to original format
        if isinstance(masks, np.ndarray) and masks.ndim == 3:
            # Return the result as-is (this includes new objects, so more channels)
            return result_masks
        elif isinstance(masks, np.ndarray) and masks.ndim == 2:
            # For single mask, return the first channel if available
            if isinstance(result_masks, np.ndarray) and result_masks.ndim == 3:
                return result_masks[..., 0]
            return result_masks
        
        return result_masks

    def apply_to_bboxes(self, bboxes, alpha, paste_objects, **params):
        """Apply copy-paste to bounding boxes."""
        if alpha is None or len(paste_objects) == 0:
            return bboxes
        
        # Since the original mask is not available in params, we need to reconstruct
        # the bounding boxes from the paste_objects directly
        updated_bboxes = []
        
        # Add original bboxes first (assume they're still valid for now)
        # In a more sophisticated implementation, we'd check overlap with alpha mask
        for bbox in bboxes:
            updated_bboxes.append(bbox)
        
        # Extract bboxes from the paste objects directly
        for i, paste_obj in enumerate(paste_objects):
            paste_mask = paste_obj['mask']
            shift_x = paste_obj['shift_x']
            shift_y = paste_obj['shift_y']
            
            # Apply the shift to get the final position
            h, w = paste_mask.shape
            shifted_mask = np.zeros_like(paste_mask)
            
            if shift_x != 0 or shift_y != 0:
                # Calculate source and destination bounds
                src_y_start = max(0, -shift_y)
                src_y_end = min(h, h - shift_y)
                src_x_start = max(0, -shift_x)
                src_x_end = min(w, w - shift_x)
                
                dst_y_start = max(0, shift_y)
                dst_y_end = dst_y_start + (src_y_end - src_y_start)
                dst_x_start = max(0, shift_x)
                dst_x_end = dst_x_start + (src_x_end - src_x_start)
                
                # Copy the shifted region
                shifted_mask[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                    paste_mask[src_y_start:src_y_end, src_x_start:src_x_end]
            else:
                shifted_mask = paste_mask.copy()
            
            # Extract bbox from the shifted mask
            rows = np.any(shifted_mask, axis=1)
            cols = np.any(shifted_mask, axis=0)
            
            if rows.any() and cols.any():
                y_indices = np.where(rows)[0]
                x_indices = np.where(cols)[0]
                
                x1, x2 = x_indices[[0, -1]]
                y1, y2 = y_indices[[0, -1]]
                
                # Ensure bbox is within image bounds and valid
                h, w = shifted_mask.shape
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(x1+1, min(x2 + 1, w))
                y2 = max(y1+1, min(y2 + 1, h))
                
                # Create new bbox for pasted object (ensure it's a list, not tuple)
                # Format: [x1, y1, x2, y2] for pascal_voc (labels handled separately)
                new_bbox = [float(x1), float(y1), float(x2), float(y2)]
                updated_bboxes.append(new_bbox)
                print(f"DEBUG: Added pasted bbox: {new_bbox}")
        
        print(f"DEBUG: Returning {len(updated_bboxes)} bboxes to Albumentations")
        return updated_bboxes

    def apply_to_labels(self, labels, alpha, paste_objects, **params):
        """Apply copy-paste to labels - add labels for pasted objects."""
        if alpha is None or len(paste_objects) == 0:
            return labels
        
        # Start with original labels
        updated_labels = list(labels)
        
        # Add labels for each pasted object
        for paste_obj in paste_objects:
            # Add label 2 for pasted objects (you can modify this logic)
            updated_labels.append(2)
        
        print(f"DEBUG: Original labels: {labels}, Updated labels: {updated_labels}")
        return updated_labels

    def get_transform_init_args_names(self):
        return (
            "blend",
            "sigma",
            "pct_objects_paste",
            "max_paste_objects"
        )

# Note: The copy_paste_class decorator is no longer needed as this implementation
# works out of the box with standard Albumentations data format.