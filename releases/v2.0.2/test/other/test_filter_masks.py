import numpy as np

def filter_empty_masks(sample):
    # Compute a mask indicating whether each channel is empty
    is_empty = np.all(sample == 0, axis=(0, 1))

    # Find the indices of non-empty channels
    kept_indices = np.where(~is_empty)[0]

    # Filter the non-empty channels
    sample = sample[..., kept_indices]

    if sample.shape[-1] == 0:
        # If all channels were empty, add an all-zero channel
        sample = np.zeros(sample.shape[:-1] + (1,), dtype=sample.dtype)

    return sample, kept_indices


mask_1 = np.zeros((4, 256, 256))
mask_1[0, :5, :5] = 1
mask_1[1, 7:10, 8:9] = 1
mask_1[2, 9:10, 9:10] = 1
mask_1[3, :4, 9:10] = 1
mask_1 = np.transpose(mask_1, (1, 2, 0))


mask, idx = filter_empty_masks(mask_1)
print(mask_1.shape)
print(mask.shape)
print(idx)