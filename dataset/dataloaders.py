import torch
import random
import numpy as np
import torch.utils.data

from typing import Any, Callable, List, Optional
import torch.utils
from torch.utils.data import DataLoader
from torch.utils.data import Sampler, DistributedSampler


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch

def empty_collate_fn(batch):
    """
    Empty collate_fn to filter out None items in a batch.
    """
    batch = [item for item in batch if item['labels'].shape[0] > 0]
    if not batch: 
        return None
    
    return trivial_batch_collator(batch)

def metadata_collate_fn(batch):
    print(type(batch[0]))
    print(batch[0].keys())
    batch = [item for item in batch if "metadata" not in item]
    return trivial_batch_collator(batch)

def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int = 1,
    num_workers: int = 0,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    sampler: Optional[Sampler] = None,
    seed: Optional[int] = None,
    shuffle: bool = False,
    distributed: bool = False,
) -> DataLoader:
    """
    Builds a data loader for either normal or distributed training.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to load.
        batch_size (int): Number of samples per batch to load.
        num_workers (int): How many subprocesses to use for data loading.
        collate_fn (Callable, optional): Function to merge a list of samples into a mini-batch.
        sampler (torch.utils.data.Sampler, optional): Sampler to use for loading data.
        seed (int, optional): Seed for random number generators.
        rank (int): Rank of the current process in distributed settings.
        world_size (int): Total number of processes in distributed settings.
        distributed (bool): Whether to use distributed data loading.

    Returns:
        torch.utils.data.DataLoader: A data loader for the specified dataset.
    """
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True)
        shuffle = False
    else:
        shuffle = shuffle if sampler is None else False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        shuffle=shuffle,
        pin_memory=True,
    )
