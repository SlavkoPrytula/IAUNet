from ast import Num
from typing import Any, Callable, List, Optional
import torch.utils
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Sampler, RandomSampler, SequentialSampler, DistributedSampler
import torch
import random
import numpy as np
from functools import partial

import torch.utils.data
import torch.utils.data.sampler


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


# def empty_collate_fn(batch):
#     """
#     Empty collate_fn to filter out None items in a batch.
#     """
#     batch = [item for item in batch if item is not None]
#     if not batch: 
#         return None
    
#     return trivial_batch_collator(batch)


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



# Based on https://github.com/CaoWGG/multi-scale-training
class BatchSampler(object):
    def __init__(self, sampler, batch_size, drop_last, multiscale_step=None, img_sizes=None):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        if multiscale_step is not None and multiscale_step < 1 :
            raise ValueError("multiscale_step should be > 0, but got "
                             "multiscale_step={}".format(multiscale_step))
        if multiscale_step is not None and img_sizes is None:
            raise ValueError("img_sizes must a list, but got img_sizes={} ".format(img_sizes))

        self.multiscale_step = multiscale_step
        self.img_sizes = img_sizes

    def __iter__(self):
        num_batch = 0
        batch = []
        size = (512, 512)
        for idx in self.sampler:
            batch.append([idx,size])
            if len(batch) == self.batch_size:
                yield batch
                num_batch+=1
                batch = []
                if self.multiscale_step and num_batch % self.multiscale_step == 0 :
                    size = random.choice(self.img_sizes)
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size   
        


def build_loader_ms(
    dataset: Dataset,
    *,
    batch_size: int = 1,
    num_workers: int = 0,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
) -> DataLoader:
    """
    Similar to `build_detection_train_loader`, with default batch size = 1,
    and sampler = :class:`InferenceSampler`. This sampler coordinates all workers
    to produce the exact set of all samples.

    Args:
        dataset: nn.Dataset class,
            or a pytorch dataset. They can be obtained
            by using :func:`DatasetCatalog.get`.
        batch_size: the batch size of the data loader to be created.
            Default to 1 image per worker since this is the standard when reporting
            inference time in papers.
        num_workers: number of parallel data loading workers
        collate_fn: same as the argument of `torch.utils.data.DataLoader`.
            Defaults to do no collation and return a list of data.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))
    """
    
    return DataLoader(
        dataset,
        # batch_size=batch_size,
        # drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
        pin_memory=True, 
        batch_sampler=BatchSampler(
            RandomSampler(dataset),
            batch_size=batch_size,
            multiscale_step=1,
            drop_last=True,
            img_sizes=[(448, 448), (512, 512), (640, 640), (704, 704), (768, 768)]
            ),
    )



# def build_loader(
#     dataset: Dataset,
#     *,
#     batch_size: int = 1,
#     num_workers: int = 0,
#     collate_fn: Optional[Callable[[List[Any]], Any]] = None,
#     sampler = torch.utils.data.Sampler,
#     seed: Optional[int] = None,
#     rank: int = 0
# ) -> DataLoader:
#     """
#     Similar to `build_detection_train_loader`, with default batch size = 1,
#     and sampler = :class:`InferenceSampler`. This sampler coordinates all workers
#     to produce the exact set of all samples.

#     Args:
#         dataset: nn.Dataset class,
#             or a pytorch dataset. They can be obtained
#             by using :func:`DatasetCatalog.get`.
#         batch_size: the batch size of the data loader to be created.
#             Default to 1 image per worker since this is the standard when reporting
#             inference time in papers.
#         num_workers: number of parallel data loading workers
#         collate_fn: same as the argument of `torch.utils.data.DataLoader`.
#             Defaults to do no collation and return a list of data.
#         seed: optional seed for deterministic behavior
#         rank: the rank of the current process (useful in distributed settings)

#     Returns:
#         DataLoader: a torch DataLoader, that loads the given detection
#         dataset, with test-time transformation and batching.

#     Examples:
#     ::
#         data_loader = build_detection_test_loader(
#             DatasetRegistry.get("my_test"),
#             mapper=DatasetMapper(...))
#     """
#     # init_fn = partial(
#     #     worker_init_fn, num_workers=num_workers, rank=rank,
#     #     seed=seed) if seed is not None else None
#     # generator = torch.Generator().manual_seed(seed)
    

#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         drop_last=False,
#         num_workers=num_workers,
#         collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
#         pin_memory=True, 
#         # worker_init_fn=init_fn,
#         # sampler=sampler,
#         # generator=generator
#     )



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
