import os
import os.path as osp
import tempfile
import pickle
import torch
import torch.distributed as dist
from typing import Callable, Optional, Tuple, Union, List, Any, Dict, Sequence, Iterable, Mapping
from abc import ABCMeta, abstractmethod
import shutil
from itertools import zip_longest, chain
from torch import distributed as torch_dist
from torch.distributed import ProcessGroup
from torch import Tensor

TORCH_VERSION = torch.__version__

import warnings
from packaging.version import parse


def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Defaults to 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    assert 'parrots' not in version_str
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f'unknown prerelease version {version.pre[0]}, '
                              'version checking may go wrong')
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])  # type: ignore
    else:
        release.extend([0, 0])
    return tuple(release)


def is_distributed() -> bool:
    """Check if distributed training is available and initialized."""
    return dist.is_available() and dist.is_initialized()


def get_dist_info():
    """Get the rank and world size in distributed training."""
    if is_distributed():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size



def broadcast(data: Tensor,
              src: int = 0,
              group: Optional[ProcessGroup] = None) -> None:
    """Broadcast the data from ``src`` process to the whole group.

    ``data`` must have the same number of elements in all processes
    participating in the collective.

    Note:
        Calling ``broadcast`` in non-distributed environment does nothing.

    Args:
        data (Tensor): Data to be sent if ``src`` is the rank of current
            process, and data to be used to save received data otherwise.
        src (int): Source rank. Defaults to 0.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Examples:
        >>> import torch
        >>> import mmengine.dist as dist

        >>> # non-distributed environment
        >>> data = torch.arange(2, dtype=torch.int64)
        >>> data
        tensor([0, 1])
        >>> dist.broadcast(data)
        >>> data
        tensor([0, 1])

        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> data = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> data
        tensor([1, 2]) # Rank 0
        tensor([3, 4]) # Rank 1
        >>> dist.broadcast(data)
        >>> data
        tensor([1, 2]) # Rank 0
        tensor([1, 2]) # Rank 1
    """
    if get_world_size(group) > 1:
        if group is None:
            group = get_default_group()

        input_device = get_data_device(data)
        backend_device = get_comm_device(group)
        data_on_device = cast_data_device(data, backend_device)
        # broadcast requires tensor is contiguous
        data_on_device = data_on_device.contiguous()  # type: ignore
        torch_dist.broadcast(data_on_device, src, group)

        if get_rank(group) != src:
            cast_data_device(data_on_device, input_device, data)


def get_data_device(data: Union[Tensor, Mapping, Iterable]) -> torch.device:
    if isinstance(data, Tensor):
        return data.device
    elif isinstance(data, Mapping):
        pre = None
        for v in data.values():
            cur = get_data_device(v)
            if pre is None:
                pre = cur
            else:
                if cur != pre:
                    raise ValueError(
                        'device type in data should be consistent, but got '
                        f'{cur} and {pre}')
        if pre is None:
            raise ValueError('data should not be empty.')
        return pre
    elif isinstance(data, Iterable) and not isinstance(data, str):
        pre = None
        for item in data:
            cur = get_data_device(item)
            if pre is None:
                pre = cur
            else:
                if cur != pre:
                    raise ValueError(
                        'device type in data should be consistent, but got '
                        f'{cur} and {pre}')
        if pre is None:
            raise ValueError('data should not be empty.')
        return pre
    else:
        raise TypeError('data should be a Tensor, sequence of tensor or dict, '
                        f'but got {data}')


def get_comm_device(group: Optional[ProcessGroup] = None) -> torch.device:
    backend = get_backend(group)
    if backend == 'hccl':
        import torch_npu  # noqa: F401
        return torch.device('npu', torch.npu.current_device())
    elif backend == torch_dist.Backend.NCCL:
        return torch.device('cuda', torch.cuda.current_device())
    elif backend == 'cncl':
        import torch_mlu  # noqa: F401
        return torch.device('mlu', torch.mlu.current_device())
    elif backend == 'smddp':
        return torch.device('cuda', torch.cuda.current_device())
    elif backend == 'mccl':
        import torch_musa
        return torch.device('musa', torch_musa.current_device())
    else:
        # GLOO and MPI backends use cpu device by default
        return torch.device('cpu')


def cast_data_device(
    data: Union[Tensor, Mapping, Iterable],
    device: torch.device,
    out: Optional[Union[Tensor, Mapping, Iterable]] = None
) -> Union[Tensor, Mapping, Iterable]:
    """Recursively convert Tensor in ``data`` to ``device``.

    If ``data`` has already on the ``device``, it will not be casted again.

    Args:
        data (Tensor or list or dict): Inputs to be casted.
        device (torch.device): Destination device type.
        out (Tensor or list or dict, optional): If ``out`` is specified, its
            value will be equal to ``data``. Defaults to None.

    Returns:
        Tensor or list or dict: ``data`` was casted to ``device``.
    """
    if out is not None:
        if type(data) != type(out):
            raise TypeError(
                'out should be the same type with data, but got data is '
                f'{type(data)} and out is {type(data)}')

        if isinstance(out, set):
            raise TypeError('out should not be a set')

    if isinstance(data, Tensor):
        if get_data_device(data) == device:
            data_on_device = data
        else:
            data_on_device = data.to(device)

        if out is not None:
            # modify the value of out inplace
            out.copy_(data_on_device)  # type: ignore

        return data_on_device
    elif isinstance(data, Mapping):
        data_on_device = {}
        if out is not None:
            data_len = len(data)
            out_len = len(out)  # type: ignore
            if data_len != out_len:
                raise ValueError('length of data and out should be same, '
                                 f'but got {data_len} and {out_len}')

            for k, v in data.items():
                data_on_device[k] = cast_data_device(v, device,
                                                     out[k])  # type: ignore
        else:
            for k, v in data.items():
                data_on_device[k] = cast_data_device(v, device)

        if len(data_on_device) == 0:
            raise ValueError('data should not be empty')

        # To ensure the type of output as same as input, we use `type(data)`
        # to wrap the output
        return type(data)(data_on_device)  # type: ignore
    elif isinstance(data, Iterable) and not isinstance(
            data, str) and not isinstance(data, np.ndarray):
        data_on_device = []
        if out is not None:
            for v1, v2 in zip(data, out):
                data_on_device.append(cast_data_device(v1, device, v2))
        else:
            for v in data:
                data_on_device.append(cast_data_device(v, device))

        if len(data_on_device) == 0:
            raise ValueError('data should not be empty')

        return type(data)(data_on_device)  # type: ignore
    else:
        raise TypeError('data should be a Tensor, list of tensor or dict, '
                        f'but got {data}')

                        
def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def collect_results_cpu(result_part: list, size: int, tmpdir: Optional[str] = None) -> Optional[list]:
    """Collect results under CPU mode."""
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part[:size]

    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ), 32, dtype=torch.uint8)
        if rank == 0:
            mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8)
            dir_tensor[:len(tmpdir)] = tmpdir
        broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.numpy().tobytes().decode().rstrip()
    else:
        mkdir_or_exist(tmpdir)

    # dump the part result to the dir
    with open(osp.join(tmpdir, f'part_{rank}.pkl'), 'wb') as f:  # type: ignore
        pickle.dump(result_part, f, protocol=2)

    barrier()

    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            path = osp.join(tmpdir, f'part_{i}.pkl')  # type: ignore
            if not osp.exists(path):
                raise FileNotFoundError(
                    f'{tmpdir} is not an shared directory for '
                    f'rank {i}, please make sure {tmpdir} is a shared '
                    'directory for all ranks!')
            with open(path, 'rb') as f:
                part_list.append(pickle.load(f))
        # sort the results
        ordered_results = []
        zipped_results = zip_longest(*part_list)
        ordered_results = [
            i for i in chain.from_iterable(zipped_results) if i is not None
        ]
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)  # type: ignore
        return ordered_results


def collect_results_gpu(result_part: list, size: int) -> Optional[list]:
    """Collect results under GPU mode."""
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part[:size]

    part_list = dist.all_gather_object(result_part)

    if rank == 0:
        ordered_results = [i for sublist in part_list for i in sublist][:size]
        return ordered_results
    else:
        return None


def collect_results(results: list, size: int, device: str = 'cpu', tmpdir: Optional[str] = None) -> Optional[list]:
    """Collected results in distributed environments."""
    if device not in ['gpu', 'cpu']:
        raise NotImplementedError(f"Device must be 'cpu' or 'gpu', but got {device}")

    if device == 'gpu':
        return collect_results_gpu(results, size)
    else:
        return collect_results_cpu(results, size, tmpdir)


def is_main_process() -> bool:
    """Check if the current process is the main process."""
    rank, _ = get_dist_info()
    return rank == 0



def _tensor_to_object(tensor: Tensor, tensor_size: int) -> Any:
    """Deserialize tensor to picklable python object."""
    buf = tensor.cpu().numpy().tobytes()[:tensor_size]
    return pickle.loads(buf)


def _object_to_tensor(obj: Any) -> Tuple[Tensor, Tensor]:
    """Serialize picklable python object to tensor."""
    byte_storage = torch.ByteStorage.from_buffer(pickle.dumps(obj))
    # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor
    # and specifying dtype. Otherwise, it will cause 100X slowdown.
    # See: https://github.com/pytorch/pytorch/issues/65696
    byte_tensor = torch.ByteTensor(byte_storage)
    local_size = torch.LongTensor([byte_tensor.numel()])
    return byte_tensor, local_size


def _broadcast_object_list(object_list: List[Any],
                           src: int = 0,
                           group: Optional[ProcessGroup] = None) -> None:
    """Broadcast picklable objects in ``object_list`` to the whole group.

    Similar to :func:`broadcast`, but Python objects can be passed in. Note
    that all objects in ``object_list`` must be picklable in order to be
    broadcasted.
    """
    if torch_dist.distributed_c10d._rank_not_in_group(group):
        return

    my_rank = get_rank()
    # Serialize object_list elements to tensors on src rank.
    if my_rank == src:
        tensor_list, size_list = zip(
            *[_object_to_tensor(obj) for obj in object_list])
        object_sizes_tensor = torch.cat(size_list)
    else:
        object_sizes_tensor = torch.empty(len(object_list), dtype=torch.long)

    # Current device selection.
    # To preserve backwards compatibility, ``device`` is ``None`` by default.
    # in which case we run current logic of device selection, i.e.
    # ``current_device`` is CUDA if backend is NCCL otherwise CPU device. In
    # the case it is not ``None`` we move the size and object tensors to be
    # broadcasted to this device.
    group_backend = get_backend(group)
    is_nccl_backend = group_backend == torch_dist.Backend.NCCL
    current_device = torch.device('cpu')
    is_hccl_backend = group_backend == 'hccl'
    is_cncl_backend = group_backend == 'cncl'
    is_mccl_backend = group_backend == 'mccl'
    if is_hccl_backend:
        current_device = torch.device('npu', torch.npu.current_device())
        object_sizes_tensor = object_sizes_tensor.to(current_device)
    elif is_cncl_backend:
        current_device = torch.device('mlu', torch.mlu.current_device())
        object_sizes_tensor = object_sizes_tensor.to(current_device)
    elif is_mccl_backend:
        current_device = torch.device('musa', torch.musa.current_device())
        object_sizes_tensor = object_sizes_tensor.to(current_device)
    elif is_nccl_backend:
        # See note about using torch.cuda.current_device() here in
        # docstring. We cannot simply use my_rank since rank == device is
        # not necessarily true.
        current_device = torch.device('cuda', torch.cuda.current_device())
        object_sizes_tensor = object_sizes_tensor.to(current_device)

    # Broadcast object sizes
    torch_dist.broadcast(object_sizes_tensor, src=src, group=group)

    # Concatenate and broadcast serialized object tensors
    if my_rank == src:
        object_tensor = torch.cat(tensor_list)
    else:
        object_tensor = torch.empty(
            torch.sum(object_sizes_tensor).int().item(),
            dtype=torch.uint8,
        )

    if is_nccl_backend or is_hccl_backend or is_cncl_backend:
        object_tensor = object_tensor.to(current_device)
    torch_dist.broadcast(object_tensor, src=src, group=group)
    # Deserialize objects using their stored sizes.
    offset = 0
    if my_rank != src:
        for i, obj_size in enumerate(object_sizes_tensor):
            obj_view = object_tensor[offset:offset + obj_size]
            obj_view = obj_view.type(torch.uint8)
            if obj_view.device != torch.device('cpu'):
                obj_view = obj_view.cpu()
            offset += obj_size
            object_list[i] = _tensor_to_object(obj_view, obj_size)



def get_default_group() -> Optional[ProcessGroup]:
    """Return default process group."""
    return torch_dist.distributed_c10d._get_default_group()


def barrier(group: Optional[ProcessGroup] = None) -> None:
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        torch_dist.barrier(group)


def get_backend(group: Optional[ProcessGroup] = None) -> Optional[str]:
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_backend(group)
    else:
        return None


def get_rank(group: Optional[ProcessGroup] = None) -> int:
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_rank(group)
    else:
        return 0
    
def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_world_size(group)
    else:
        return 1
    
def broadcast_object_list(data: List[Any],
                          src: int = 0,
                          group: Optional[ProcessGroup] = None) -> None:
    assert isinstance(data, list)

    if get_world_size(group) > 1:
        if group is None:
            group = get_default_group()

        if digit_version(TORCH_VERSION) >= digit_version('1.8.0'):
            torch_dist.broadcast_object_list(data, src, group)
        else:
            _broadcast_object_list(data, src, group)


def _to_cpu(data: Any) -> Any:
    """Move data to CPU."""
    if isinstance(data, torch.Tensor):
        return data.cpu()
    elif isinstance(data, list):
        return [_to_cpu(d) for d in data]
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    else:
        return data


class BaseMetric(metaclass=ABCMeta):
    """Base class for a metric.

    The metric first processes each batch of data_samples and predictions,
    and appends the processed results to the results list. Then it
    collects all results together from all ranks if distributed training
    is used. Finally, it computes the metrics of the entire dataset.

    A subclass of class:`BaseMetric` should assign a meaningful value to the
    class attribute `default_prefix`. See the argument `prefix` for details.

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
        collect_dir: (str, optional): Synchronize directory for collecting data
            from different ranks. This argument should only be configured when
            ``collect_device`` is 'cpu'. Defaults to None.
            `New in version 0.7.3.`
    """

    default_prefix: Optional[str] = None

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 collect_dir: Optional[str] = None) -> None:
        if collect_dir is not None and collect_device != 'cpu':
            raise ValueError('`collec_dir` could only be configured when '
                             "`collect_device='cpu'`")

        self._dataset_meta: Union[None, dict] = None
        self.collect_device = collect_device
        self.results: List[Any] = []
        self.prefix = prefix or self.default_prefix
        self.collect_dir = collect_dir

        # if self.prefix is None:
        #     print_log(
        #         'The prefix is not set in metric class '
        #         f'{self.__class__.__name__}.',
        #         logger='current',
        #         level=logging.WARNING)

    @property
    def dataset_meta(self) -> Optional[dict]:
        """Optional[dict]: Meta info of the dataset."""
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set the dataset meta info to the metric."""
        self._dataset_meta = dataset_meta

    @abstractmethod
    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Any): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """

    @abstractmethod
    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

    # def evaluate(self) -> dict:
    #     # compute coco metrics
    #     eval_results = self.compute_metrics(self.results)
    #     # reset the results list
    #     self.results.clear()

    #     return eval_results

    def evaluate(self, size: int) -> dict:
        if len(self.results) == 0:
            print('No results to evaluate.')

        if self.collect_device == 'cpu':
            results = collect_results(
                self.results,
                size,
                self.collect_device,
                tmpdir=self.collect_dir)
        else:
            results = collect_results(self.results, size, self.collect_device)

        if is_main_process():
            # cast all tensors in results list to cpu
            results = _to_cpu(results)
            _metrics = self.compute_metrics(results)  # type: ignore
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        print(f'metrics from base: {metrics}')
        return metrics[0]

