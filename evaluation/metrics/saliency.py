import numpy as np
from skimage.transform import resize
import random

import torch.nn.functional as F

from sklearn.metrics import matthews_corrcoef


def normalize(x, method='standard', axis=None):
    '''Normalizes the input with specified method.
    Parameters
    ----------
    x : array-like
    method : string, optional
        Valid values for method are:
        - 'standard': mean=0, std=1
        - 'range': min=0, max=1
        - 'sum': sum=1
    axis : int, optional
        Axis perpendicular to which array is sliced and normalized.
        If None, array is flattened and normalized.
    Returns
    -------
    res : numpy.ndarray
        Normalized array.
    '''
    # TODO: Prevent divided by zero if the map is flat
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res



def NSS(saliency_map, fixation_map):
	'''
	Normalized scanpath saliency of a saliency map,
	defined as the mean value of normalized (i.e., standardized) saliency map at fixation locations.
	You can think of it as a z-score. (Larger value implies better performance.)
	Parameters
	----------
	saliency_map : real-valued matrix
		If the two maps are different in shape, saliency_map will be resized to match fixation_map..
	fixation_map : binary matrix
		Human fixation map (1 for fixated location, 0 for elsewhere).
	Returns
	-------
	NSS : float, positive
	'''
	s_map = np.array(saliency_map, copy=False)
	f_map = np.array(fixation_map, copy=False) > 0.5
	if s_map.shape != f_map.shape:
		s_map = resize(s_map, f_map.shape)
	# Normalize saliency map to have zero mean and unit std
	s_map = normalize(s_map, method='standard')
	# Mean saliency value at fixation locations
	return np.mean(s_map[f_map])


def AUC_Borji(saliency_map, fixation_map, n_rep=100, step_size=0.1, rand_sampler=None):
	'''
	This measures how well the saliency map of an image predicts the ground truth human fixations on the image.
	ROC curve created by sweeping through threshold values at fixed step size
	until the maximum saliency map value.
	True positive (tp) rate correspond to the ratio of saliency map values above threshold
	at fixation locations to the total number of fixation locations.
	False positive (fp) rate correspond to the ratio of saliency map values above threshold
	at random locations to the total number of random locations
	(as many random locations as fixations, sampled uniformly from fixation_map ALL IMAGE PIXELS),
	averaging over n_rep number of selections of random locations.
	Parameters
	----------
	saliency_map : real-valued matrix
	fixation_map : binary matrix
		Human fixation map.
	n_rep : int, optional
		Number of repeats for random sampling of non-fixated locations.
	step_size : int, optional
		Step size for sweeping through saliency map.
	rand_sampler : callable
		S_rand = rand_sampler(S, F, n_rep, n_fix)
		Sample the saliency map at random locations to estimate false positive.
		Return the sampled saliency values, S_rand.shape=(n_fix,n_rep)
	Returns
	-------
	AUC : float, between [0,1]
	'''
	saliency_map = np.array(saliency_map, copy=False)
	fixation_map = np.array(fixation_map, copy=False) > 0.5
	# If there are no fixation to predict, return NaN
	if not np.any(fixation_map):
		print('no fixation to predict')
		return np.nan
	# Make the saliency_map the size of the fixation_map
	if saliency_map.shape != fixation_map.shape:
		saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='nearest')
	# Normalize saliency map to have values between [0,1]
	saliency_map = normalize(saliency_map, method='range')

	S = saliency_map.ravel()
	F = fixation_map.ravel()
	S_fix = S[F] # Saliency map values at fixation locations
	n_fix = len(S_fix)
	n_pixels = len(S)
	# For each fixation, sample n_rep values from anywhere on the saliency map
	if rand_sampler is None:
		r = random.randint(0, n_pixels, [n_fix, n_rep])
		S_rand = S[r] # Saliency map values at random locations (including fixated locations!? underestimated)
	else:
		S_rand = rand_sampler(S, F, n_rep, n_fix)
	# Calculate AUC per random split (set of random locations)
	auc = np.zeros(n_rep) * np.nan
	for rep in range(n_rep):
		thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:,rep]]):step_size][::-1]
		tp = np.zeros(len(thresholds)+2)
		fp = np.zeros(len(thresholds)+2)
		tp[0] = 0; tp[-1] = 1
		fp[0] = 0; fp[-1] = 1
		for k, thresh in enumerate(thresholds):
			tp[k+1] = np.sum(S_fix >= thresh) / float(n_fix)
			fp[k+1] = np.sum(S_rand[:,rep] >= thresh) / float(n_fix)
		auc[rep] = np.trapz(tp, fp)
	return np.mean(auc) # Average across random splits



def CrossEntropy(saliency_map, fixation_map):
    # saliency_map /= saliency_map.sum()

    # Avoid log(0) for zero values in saliency map by adding a small constant (epsilon)
    epsilon = 1e-10
    saliency_map = np.clip(saliency_map, epsilon, 1 - epsilon)
    fixation_map = np.clip(fixation_map, epsilon, 1 - epsilon)

    # Calculate cross-entropy
    cross_entropy = -(fixation_map * np.log(saliency_map) + (1 - fixation_map) * np.log(1 - saliency_map))
    return cross_entropy.mean()


def MCC(saliency_map, fixation_map, threshold=0.5):
	"""
	Compute the Matthews Correlation Coefficient between a ground truth binary mask
	and an activation map.

	Args:
	ground_truth (numpy.ndarray): Ground truth binary mask.
	activation_map (numpy.ndarray): Activation map (same shape as ground_truth).
	threshold (float): Threshold to apply to the activation map to get binary predictions.

	Returns:
	float: The Matthews Correlation Coefficient.
	"""
	# Apply threshold to the activation map
	saliency_map = (saliency_map > threshold).astype(int)
	ground_truth_flat = (fixation_map > threshold).astype(int)

	# Flatten the arrays for MCC calculation
	ground_truth_flat = fixation_map.flatten()
	saliency_map_flat = saliency_map.flatten()

	# Compute MCC
	mcc = matthews_corrcoef(ground_truth_flat, saliency_map_flat)
	return mcc