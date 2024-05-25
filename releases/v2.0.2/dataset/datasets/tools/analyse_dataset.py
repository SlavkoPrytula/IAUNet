import sys
sys.path.append("./")

from dataset.datasets.brightfiled import Brightfield_Dataset
from utils.augmentations import train_transforms, valid_transforms
from configs import cfg
from tqdm import tqdm
import numpy as np


dataset = Brightfield_Dataset(cfg, dataset_type="train",
                              normalization=False,
                              transform=False)

means = []
stds = []
for data in tqdm(dataset):
    image = data['image'].float().numpy()
    image = np.transpose(image, (1, 2, 0))

    mean_per_channel = np.mean(image.reshape(-1, 3), axis=0)
    std_per_channel = np.std(image.reshape(-1, 3), axis=0)

    means.append(mean_per_channel)
    stds.append(std_per_channel)
    # print("="*10)

# print(f"Means: {means}")
# print(f"Stds: {stds}")
# print()

mean = np.mean(means, axis=0)
std = np.mean(stds, axis=0)
print(f"Mean: {mean}")
print(f"Std: {std}")