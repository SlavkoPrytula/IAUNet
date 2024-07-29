import numpy as np

normalize_0_255 = lambda a: (a * 255) / a.max()


def normalize_mean_std(a, axis=None):
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.std(a, axis=axis, keepdims=True)
    return (a - mean) / std


def normalize_max(a):
    return a / a.max()


def normalize_255_mean_std(a):
    mean = np.array([0.7720342, 0.74582646, 0.76392896])
    std = np.array([0.24745085, 0.26182273, 0.25782376])
    return (a - mean) / std


def normalize_255_mean_std_v2(a):
    mean = np.array([0.485, 0.456]), #0.406])
    std = np.array([0.229, 0.224]), #0.225])
    return (a - mean) / std


def normalize_custom(a):
    a = normalize_max(a) # 0-1
    # a = normalize_255_mean_std_v2(a)
    return a



# normalize = lambda a: normalize_255_mean_std_v2(a)
normalize = lambda a: normalize_mean_std(a)
# normalize = lambda a: normalize_custom(a)