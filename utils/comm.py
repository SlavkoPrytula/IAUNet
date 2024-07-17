import os
import functools
import numpy as np
import torch
import torch.distributed as dist


_LOCAL_PROCESS_GROUP = None



def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return 1

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    return get_rank() == 0



@functools.lru_cache()
def create_local_process_group(num_workers_per_machine: int) -> None:
    """
    Create a process group that contains ranks within the same machine.

    Detectron2's launch() in engine/launch.py will call this function. If you start
    workers without launch(), you'll have to also call this. Otherwise utilities
    like `get_local_rank()` will not work.

    This function contains a barrier. All processes must call it together.

    Args:
        num_workers_per_machine: the number of worker processes per machine. Typically
          the number of GPUs.
    """
    global _LOCAL_PROCESS_GROUP
    assert _LOCAL_PROCESS_GROUP is None
    assert get_world_size() % num_workers_per_machine == 0
    num_machines = get_world_size() // num_workers_per_machine
    machine_rank = get_rank() // num_workers_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_workers_per_machine, (i + 1) * num_workers_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            _LOCAL_PROCESS_GROUP = pg


def get_local_process_group():
    """
    Returns:
        A torch process group which only includes processes that are on the same
        machine as the current process. This group can be useful for communication
        within a machine, e.g. a per-machine SyncBN.
    """
    return _LOCAL_PROCESS_GROUP


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    if dist.get_backend() == dist.Backend.NCCL:
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()  # use CPU group by default, to reduce GPU RAM usage.
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)
    return output



def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return [data]
    rank = dist.get_rank(group=group)

    if rank == dst:
        output = [None for _ in range(world_size)]
        dist.gather_object(data, output, dst=dst, group=group)
        return output
    else:
        dist.gather_object(data, None, dst=dst, group=group)
        return []


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
        If workers need a shared RNG, they can use this shared seed to
        create one.

    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2**31)
    all_ints = all_gather(ints)
    return all_ints[0]


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.

    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum

    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict



# def broadcast_object_list(data,
#                           src=0,
#                           group=None) -> None:
#     """Broadcasts picklable objects in ``object_list`` to the whole group.
#     Similar to :func:`broadcast`, but Python objects can be passed in. Note
#     that all objects in ``object_list`` must be picklable in order to be
#     broadcasted.

#     Note:
#         Calling ``broadcast_object_list`` in non-distributed environment does
#         nothing.

#     Args:
#         data (List[Any]): List of input objects to broadcast.
#             Each object must be picklable. Only objects on the ``src`` rank
#             will be broadcast, but each rank must provide lists of equal sizes.
#         src (int): Source rank from which to broadcast ``object_list``.
#         group: (ProcessGroup, optional): The process group to work on. If None,
#             the default process group will be used. Default is ``None``.
#         device (``torch.device``, optional): If not None, the objects are
#             serialized and converted to tensors which are moved to the
#             ``device`` before broadcasting. Default is ``None``.

#     Note:
#         For NCCL-based process groups, internal tensor representations of
#         objects must be moved to the GPU device before communication starts.
#         In this case, the used device is given by
#         ``torch.cuda.current_device()`` and it is the user's responsibility to
#         ensure that this is correctly set so that each rank has an individual
#         GPU, via ``torch.cuda.set_device()``.

#     Examples:
#         >>> import torch
#         >>> import mmengine.dist as dist

#         >>> # non-distributed environment
#         >>> data = ['foo', 12, {1: 2}]
#         >>> dist.broadcast_object_list(data)
#         >>> data
#         ['foo', 12, {1: 2}]

#         >>> # distributed environment
#         >>> # We have 2 process groups, 2 ranks.
#         >>> if dist.get_rank() == 0:
#         >>>     # Assumes world_size of 3.
#         >>>     data = ["foo", 12, {1: 2}]  # any picklable object
#         >>> else:
#         >>>     data = [None, None, None]
#         >>> dist.broadcast_object_list(data)
#         >>> data
#         ["foo", 12, {1: 2}]  # Rank 0
#         ["foo", 12, {1: 2}]  # Rank 1
#     """
#     assert isinstance(data, list)

#     if get_world_size(group) > 1:
#         if group is None:
#             group = get_default_group()

#         if digit_version(TORCH_VERSION) >= digit_version(
#                 '1.8.0') and not is_npu_available():
#             torch_dist.broadcast_object_list(data, src, group)
#         else:
#             _broadcast_object_list(data, src, group)