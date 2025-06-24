import pytest
from hydra import initialize, compose
from utils.augmentations import valid_transforms
from dataset.datasets import * 
from utils.registry import DATASETS
from configs import cfg as _cfg


@pytest.mark.parametrize("dataset_name", [
    "revvity_25",
    "livecell",
    "isbi2014",
])
def test_registered_dataset(dataset_name):
    with initialize(config_path="../../configs", version_base="1.3"):
        cfg: _cfg = compose(config_name="train", overrides=[f"dataset={dataset_name}"])

        # Get dataset class from the registry and instantiate it
        dataset_cls = DATASETS.get(cfg.dataset.type)
        dataset = dataset_cls(cfg, dataset_type="train", transform=valid_transforms(cfg))

        # Basic checks
        assert len(dataset) > 0, f"Dataset '{dataset_name}' is empty."

        sample = dataset[0]

        # Key presence
        assert isinstance(sample, dict), f"Sample from '{dataset_name}' is not a dictionary."
        assert "image" in sample, f"Sample missing 'image' key for '{dataset_name}'."
        assert "instance_masks" in sample, f"Sample missing 'instance_masks' key for '{dataset_name}'."
        assert "labels" in sample, f"Sample missing 'labels' key for '{dataset_name}'."

        # Shape checks
        assert sample["image"].ndim == 3, f"Image should be (C, H, W), got {sample['image'].ndim} dimensions."
        assert sample["instance_masks"].ndim == 3, f"Instance masks should be (N, H, W), got {sample['instance_masks'].ndim} dimensions."
        assert sample["image"].shape[-2:] == sample["instance_masks"].shape[-2:], f"Image and mask dimensions do not match, got {sample['image'].shape[-2:]} and {sample['instance_masks'].shape[-2:]}."
        
        # Label checks
        assert sample["labels"].ndim == 1, f"Labels should be 1D tensor, got {sample['labels'].ndim} dimensions."
        assert len(sample["labels"]) == sample["instance_masks"].shape[0], f"Number of labels doesn't match number of masks. Expected {len(sample['labels'])}, got {sample['instance_masks'].shape[0]} masks."

        # Metadata checks
        metadata_keys = ["img_id", "img_path", "ori_shape", "file_name", "coco_id"]
        for key in metadata_keys:
            assert key in sample, f"Metadata key '{key}' missing in sample from '{dataset_name}'"
        assert isinstance(sample["ori_shape"], (list, tuple)) and len(sample["ori_shape"]) == 2, f"'ori_shape' should be [H, W], got {sample['ori_shape']}"
